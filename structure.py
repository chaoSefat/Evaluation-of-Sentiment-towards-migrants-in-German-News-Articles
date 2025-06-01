#!/usr/bin/env python3
"""
Leipzig Corpus Data Processor

A modular script to process Leipzig corpus data from hierarchical folder structure
and convert it to structured JSON files organized by year.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# =============================================================================
# CONFIGURATION - Edit these variables as needed
# =============================================================================

# Language to process (e.g., 'deu' for German, 'eng' for English)
LANGUAGE = 'eng'

# Years to process (list of integers)
YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = 'INFO'

# Data directory path
DATA_DIR = 'data'

# Output directory path
OUTPUT_DIR = 'structured_data'

# =============================================================================


class CorpusProcessor:
    """Main class for processing Leipzig corpus data."""
    
    def __init__(self, lang: str, years: List[int], data_dir: str = "data", output_dir: str = "output"):
        self.language = lang
        self.years = years
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self._setup_logging()
        self._ensure_output_dir()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('corpus_processor.log')
            ]
        )
        logging.info(f"Processing language: {self.language}, years: {self.years}")
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory: {self.output_dir}")
    
    def _extract_language_and_year(self, path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Extract language and year from directory or file path."""
        # Try to extract from directory name pattern: deu_news_2010_1M
        pattern = r'([a-z]{3})_[a-z]+_(\d{4})_'
        
        # Check all parts of the path
        for part in path.parts:
            match = re.search(pattern, part)
            if match:
                lang, year = match.groups()
                return lang, year
        
        # Fallback: try to find year in any part of the path
        for part in path.parts:
            year_match = re.search(r'\b(\d{4})\b', part)
            if year_match:
                year = year_match.group(1)
                # Try to find language code
                lang_match = re.search(r'\b([a-z]{3})_', part)
                if lang_match:
                    return lang_match.group(1), year
                return None, year
        
        return None, None
    

    
    def _read_sentences_file(self, file_path: Path) -> Dict[str, str]:
        """Read sentences file and return dict mapping number to sentence."""
        sentences = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split('\t', 1)
                        if len(parts) != 2:
                            logging.warning(f"Malformed line {line_num} in {file_path}: expected 2 parts, got {len(parts)}")
                            continue
                        
                        number, sentence = parts
                        sentences[number] = sentence
                        
                    except Exception as e:
                        logging.warning(f"Error processing line {line_num} in {file_path}: {e}")
            
            logging.info(f"Read {len(sentences)} sentences from {file_path}")
            return sentences
            
        except FileNotFoundError:
            logging.error(f"Sentences file not found: {file_path}")
            return {}
        except Exception as e:
            logging.error(f"Error reading sentences file {file_path}: {e}")
            return {}
    

    
    def _process_corpus_directory(self, corpus_dir: Path) -> List[Dict]:
        """Process a single corpus directory and return list of records."""
        records = []
        
        # Extract language and year
        lang, year = self._extract_language_and_year(corpus_dir)
        
        if not lang or not year:
            logging.warning(f"Could not extract language/year from {corpus_dir}")
            return records
        
        # Skip if language doesn't match config
        if lang != self.language:
            logging.debug(f"Skipping {corpus_dir}: language {lang} doesn't match config {self.language}")
            return records
        
        # Skip if year not in config
        if int(year) not in self.years:
            logging.debug(f"Skipping {corpus_dir}: year {year} not in config years {self.years}")
            return records
        
        # Find sentences files
        sentences_pattern = f"*-sentences.txt"
        sentences_files = list(corpus_dir.glob(sentences_pattern))
        
        if not sentences_files:
            logging.warning(f"No sentences files found in {corpus_dir}")
            return records
        
        # Process each sentences file
        for sentences_file in sentences_files:
            base_name = sentences_file.stem.replace('-sentences', '')
            logging.info(f"Processing {sentences_file}")
            
            # Read sentences
            sentences = self._read_sentences_file(sentences_file)
            
            if not sentences:
                logging.warning(f"No sentences found in {sentences_file}, skipping")
                continue
            
            # Create records for ALL sentences
            for number, sentence in sentences.items():
                try:
                    record = {
                        "number": number,
                        "sentence": sentence,
                        "language": lang,
                        "year": year
                    }
                    records.append(record)
                    
                except Exception as e:
                    logging.warning(f"Error creating record for number {number}: {e}")
            
            logging.info(f"Created {len(sentences)} records from {base_name}")
        
        return records
    
    def _find_corpus_directories(self) -> List[Path]:
        """Find all corpus directories matching the pattern."""
        leipzig_dir = self.data_dir / "leipzig"
        
        if not leipzig_dir.exists():
            logging.error(f"Leipzig directory not found: {leipzig_dir}")
            return []
        
        corpus_dirs = []
        
        # Look for language directories
        for lang_dir in leipzig_dir.iterdir():
            if not lang_dir.is_dir():
                continue
            
            # Look for corpus directories within language directory
            for corpus_dir in lang_dir.iterdir():
                if corpus_dir.is_dir():
                    corpus_dirs.append(corpus_dir)
        
        logging.info(f"Found {len(corpus_dirs)} corpus directories")
        return corpus_dirs
    
    def process(self):
        """Main processing function."""
        logging.info("Starting corpus processing")
        
        # Find all corpus directories
        corpus_dirs = self._find_corpus_directories()
        
        if not corpus_dirs:
            logging.error("No corpus directories found")
            return
        
        # Process each directory and collect records by year
        records_by_year = {}
        
        for corpus_dir in corpus_dirs:
            logging.info(f"Processing directory: {corpus_dir}")
            records = self._process_corpus_directory(corpus_dir)
            
            # Group records by year
            for record in records:
                year = record['year']
                if year not in records_by_year:
                    records_by_year[year] = []
                records_by_year[year].append(record)
        
        # Write output files
        for year, records in records_by_year.items():
            output_file = self.output_dir / f"{self.language}_{year}.json"
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Wrote {len(records)} records to {output_file}")
                
            except Exception as e:
                logging.error(f"Failed to write output file {output_file}: {e}")
        
        # Summary statistics
        total_records = sum(len(records) for records in records_by_year.values())
        logging.info(f"Processing completed - Total: {total_records} records")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process Leipzig corpus data and convert to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python corpus_processor.py
  python corpus_processor.py --data-dir /path/to/data --output-dir /path/to/output
  
Configuration:
  Edit the variables at the top of this file to change language, years, etc.
        """
    )
    
    parser.add_argument('--data-dir', help=f'Path to data directory (default: {DATA_DIR})')
    parser.add_argument('--output-dir', help=f'Path to output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--language', help=f'Language code to process (default: {LANGUAGE})')
    parser.add_argument('--years', nargs='+', type=int, help=f'Years to process (default: {YEARS})')
    
    args = parser.parse_args()
    
    # Use command line args if provided, otherwise use global config
    language = args.language if args.language else LANGUAGE
    years = args.years if args.years else YEARS
    data_dir = args.data_dir if args.data_dir else DATA_DIR
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    
    # Initialize and run processor
    processor = CorpusProcessor(language, years, data_dir, output_dir)
    processor.process()


if __name__ == "__main__":
    main()