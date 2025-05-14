import pandas as pd
import argparse
import logging
import sys
from Bio import SeqIO
from multiprocessing import Pool

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def get_file_stats(file_path, sep=','):
    """
    Gets basic statistics of a CSV or TSV file.

    Args:
        file_path (str): Path to the file.
        sep (str, optional): Separator (',' for CSV, '\t' for TSV). Defaults to ','.

    Returns:
        dict: A dictionary containing file statistics (rows, cols, missing_values),
              or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path, sep=sep)
        rows, cols = df.shape
        missing_values = df.isnull().sum().sum()
        return {
            'rows': rows,
            'cols': cols,
            'missing_values': missing_values
        }
    except Exception as e:
        logger.error(f"Error getting stats for file '{file_path}': {e}")
        return None

def _process_fasta_record(record, transcript_ids):
    """
    Helper function for `extract_fasta_sequences` to process a single FASTA record.
    It finds matches between the record ID and a list of transcript IDs based on the
    longest common prefix.

    Args:
        record (SeqRecord): A sequence record from Biopython's SeqIO.
        transcript_ids (list): A list of transcript IDs to search for.

    Returns:
        dict: A dictionary where keys are matching transcript IDs and values are their sequences.
              Can be empty if no match is found.
    """
    matches = {}
    header_id = record.id
    record_sequence = str(record.seq)
    max_len = 0
    best_match_tid = None

    for tid in transcript_ids:
        match_len = 0
        # Find the length of the common prefix
        for i in range(min(len(tid), len(header_id))):
            if tid[i] == header_id[i]:
                match_len += 1
            else:
                break

        # Consider it a match if the common prefix length is greater than the current best
        if match_len > max_len:
            max_len = match_len
            best_match_tid = tid

    # If a best match was found, store the sequence
    if best_match_tid is not None:
        matches[best_match_tid] = record_sequence

    return matches

def extract_fasta_sequences(fasta_file, transcript_ids, threads=1):
    """
    Extracts FASTA sequences for given transcript IDs from a FASTA file using multiprocessing.
    It matches based on the longest common prefix of the transcript ID with the FASTA headers.

    Args:
        fasta_file (str): Path to the Trinity assembly FASTA file.
        transcript_ids (list): List of transcript IDs to extract.
        threads (int, optional): Number of threads to use for parsing. Defaults to 1.

    Returns:
        dict: A dictionary where keys are transcript IDs and values are their sequences.
              Returns an empty dictionary if no matching IDs are found or on error.
    """
    sequences = {}
    try:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        with Pool(processes=threads) as pool:
            # Process each FASTA record in parallel
            results = pool.starmap(_process_fasta_record, [(record, transcript_ids) for record in records])

        # Update the main sequences dictionary with the results from each process
        for result_dict in results:
            sequences.update(result_dict)

    except FileNotFoundError:
        logger.error(f"FASTA file '{fasta_file}' not found.")
    except Exception as e:
        logger.error(f"Error reading FASTA file '{fasta_file}': {e}")
    return sequences

def filter_deg_results(input_file, output_prefix, log2fc_cutoff, pvalue_cutoff, tool,
                        fasta_assembly=None, threads=1,
                        log2fc_col_name=None, pvalue_col_name=None,
                        use_adjusted_pvalue=False):
    """
    Filters differential gene expression results from EdgeR or DESeq2
    based on log2 fold change and p-value cutoffs. Optionally extracts
    the corresponding FASTA sequences for the filtered transcripts.

    Args:
        input_file (str): Path to the input CSV/TSV file containing DEG results.
        output_prefix (str): Prefix for the output files (e.g., 'filtered_deg').
        log2fc_cutoff (float): Absolute log2 fold change threshold for filtering.
        pvalue_cutoff (float): P-value threshold for filtering.
        tool (str): The DEG analysis tool used ('edgeR' or 'deseq2').
        fasta_assembly (str, optional): Path to the assembly FASTA file for sequence extraction. Defaults to None.
        threads (int, optional): Number of threads to use for FASTA parsing. Defaults to 1.
        log2fc_col_name (str, optional): Name of the log2 fold change column in the input file.
                                            If None, defaults to 'logFC' for edgeR and 'log2FoldChange' for DESeq2.
        pvalue_col_name (str, optional): Name of the p-value column in the input file.
                                            If None, defaults to 'FDR' (if use_adjusted_pvalue) or 'PValue' for edgeR,
                                            and 'padj' (if use_adjusted_pvalue) or 'pvalue' for DESeq2.
        use_adjusted_pvalue (bool, optional): If True, filter using the adjusted p-value column ('FDR' for edgeR, 'padj' for DESeq2). Defaults to False.
    """
    logger.info(f"Processing input file: '{input_file}'")
    filtered_output_csv = f"{output_prefix}_filtered.csv"
    filtered_output_fasta = f"{output_prefix}_filtered.fasta"

    try:
        # Determine the separator based on the file extension
        if input_file.lower().endswith('.csv'):
            sep = ','
        elif input_file.lower().endswith('.tsv') or input_file.lower().endswith('.txt'):
            sep = '\t'
        else:
            raise ValueError("Input file must be a CSV or TSV/TXT file.")

        # Read the input file into a pandas DataFrame, setting the first column as index if it's not purely numeric
        try:
            df = pd.read_csv(input_file, sep=sep, index_col=0)
        except ValueError:
            df = pd.read_csv(input_file, sep=sep, header=None)
            df = df.set_index(0)
        transcript_id_col = df.index.name
        if transcript_id_col is None:
            transcript_id_col = 'transcript_id'
            df.index.name = transcript_id_col

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        return
    except ValueError as e:
        logger.error(f"Invalid input file format: {e}")
        return
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return

    logger.info(f"Original number of genes/transcripts: {len(df)}")

    # Determine column names based on the specified tool
    if tool.lower() == 'edger':
        log2fc_col = log2fc_col_name if log2fc_col_name else 'logFC'
        pvalue_col = pvalue_col_name if pvalue_col_name else ('FDR' if use_adjusted_pvalue else 'PValue')
    elif tool.lower() == 'deseq2':
        log2fc_col = log2fc_col_name if log2fc_col_name else 'log2FoldChange'
        pvalue_col = pvalue_col_name if pvalue_col_name else ('padj' if use_adjusted_pvalue else 'pvalue')
    else:
        logger.error("Invalid tool specified. Must be 'edgeR' or 'deseq2'.")
        return

    # Check if the required columns exist
    if log2fc_col not in df.columns or pvalue_col not in df.columns:
        logger.error(f"Could not find log2FC column '{log2fc_col}' or p-value column '{pvalue_col}'.")
        logger.error(f"Available columns: {df.columns.tolist()}")
        return

    # Filter the DataFrame based on the cutoffs
    filtered_df = df[
        (abs(df[log2fc_col]) >= log2fc_cutoff) & (df[pvalue_col] <= pvalue_cutoff)
    ].copy()

    # Sort and remove duplicates (if any) based on p-value and absolute log2FC
    if not filtered_df.empty:
        filtered_df['abs_log2fc'] = abs(filtered_df[log2fc_col])
        filtered_df = filtered_df.sort_values(by=[pvalue_col, 'abs_log2fc'], ascending=[True, False])
        filtered_df = filtered_df.drop_duplicates(keep='first')
        filtered_df.drop(columns=['abs_log2fc'], errors='ignore', inplace=True)

    logger.info(f"Number of genes/transcripts after filtering (log2FC >= |{log2fc_cutoff}| and {pvalue_col} <= {pvalue_cutoff}): {len(filtered_df)}")

    # Save the filtered DEG results to a CSV file
    try:
        filtered_df.to_csv(filtered_output_csv)
        logger.info(f"Filtered results saved to '{filtered_output_csv}'")
    except Exception as e:
        logger.error(f"Error saving filtered CSV: {e}")

    # If a FASTA assembly file is provided, extract sequences for the filtered transcripts
    if fasta_assembly:
        filtered_transcript_ids = filtered_df.index.tolist()
        logger.info(f"Extracting FASTA sequences for {len(filtered_transcript_ids)} transcripts using {threads} threads.")
        if filtered_transcript_ids:
            transcript_sequences = extract_fasta_sequences(fasta_assembly, filtered_transcript_ids, threads=threads)
            if transcript_sequences:
                try:
                    with open(filtered_output_fasta, "w") as outfile:
                        for tid, seq in transcript_sequences.items():
                            outfile.write(f">{tid}\n{seq}\n")
                    logger.info(f"FASTA sequences saved to '{filtered_output_fasta}'.")
                except Exception as e:
                    logger.error(f"Error writing FASTA file: {e}")
            else:
                logger.warning("No sequences found for the filtered transcript IDs.")
        else:
            logger.info("No filtered transcripts to extract sequences for.")

    return

def main():
    """Main function to parse command-line arguments and run the filtering process."""
    parser = argparse.ArgumentParser(description="Filter DEG results (EdgeR/DESeq2) and optionally extract FASTA sequences.")
    parser.add_argument("input_file", help="Path to the input CSV/TSV file containing DEG results.")
    parser.add_argument("output_prefix", help="Prefix for the output files.")
    parser.add_argument("--log2fc", type=float, required=True, help="Absolute log2 fold change cutoff.")
    parser.add_argument("--pvalue", type=float, required=True, help="P-value cutoff.")
    parser.add_argument("--tool", type=str, required=True, choices=['edgeR', 'deseq2'], help="DEG analysis tool used.")
    parser.add_argument("--fasta_assembly", type=str, help="Path to the Trinity FASTA file for sequence extraction (optional).")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use for FASTA parsing (default: 1).")
    parser.add_argument("--log2fc_col", type=str, help="Name of the log2 fold change column (optional).")
    parser.add_argument("--pvalue_col", type=str, help="Name of the p-value column (optional).")
    parser.add_argument("--use_adjusted_pvalue", action="store_true", help="Filter by adjusted p-value if available.")
    parser.add_argument("--version", action='version', version='%(prog)s 1.0', help="Show version.")

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    logger = logging.getLogger(__name__)

    # Print a welcome message
    print("\nFold_filter (v1.0) (2025)\n"
          "\n \n"
          "######################################################################\n"
          "Created by Sinoy Johnson, Kochi, Kerala, India.\n"
          "######################################################################\n"
          "\n \n")

    args = parser.parse_args()

    # Run the main filtering function
    filter_deg_results(args.input_file, args.output_prefix, args.log2fc, args.pvalue, args.tool,
                        args.fasta_assembly, args.threads,
                        args.log2fc_col, args.pvalue_col,
                        args.use_adjusted_pvalue)

    logger.info("Script finished.")

if __name__ == "__main__":
    main()
