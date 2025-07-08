#!/usr/bin/env python3
"""
MirGeneDB Species Data Extractor

Extracts species information from MirGeneDB download page and creates a CSV file
with 3-letter codes, common names, and scientific names.
"""

import requests
import csv
import re
from bs4 import BeautifulSoup
import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

def fetch_download_page():
    """Fetch the MirGeneDB download page with error handling."""
    url = "https://mirgenedb.org/download"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"Fetching data from {url}...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching page: {e}")
        sys.exit(1)

def parse_species_name(name_text):
    """
    Parse species name text to extract common name and scientific name.
    
    Examples:
    - "Human (Homo sapiens)" → ("Human", "Homo sapiens")
    - "Laevipilina (Monoplacophora) (Laevipilina hyalina)" → ("Laevipilina (Monoplacophora)", "Laevipilina hyalina")
    - "West Indian fuzzy chiton (Acanthopleura granulata )" → ("West Indian fuzzy chiton", "Acanthopleura granulata")
    """
    name_text = name_text.strip()
    
    # Find all parentheses content
    parentheses_matches = re.findall(r'\([^)]+\)', name_text)
    
    if not parentheses_matches:
        # No parentheses found
        return name_text, ""
    
    # Last parentheses should contain scientific name
    scientific_name = parentheses_matches[-1][1:-1].strip()  # Remove parentheses and strip
    
    # Everything before the last parentheses is common name
    last_paren_pos = name_text.rfind('(')
    common_name = name_text[:last_paren_pos].strip()
    
    return common_name, scientific_name

def get_ncbi_taxid(scientific_name):
    """Get NCBI taxonomy ID for a scientific name."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'taxonomy',
        'term': scientific_name,
        'retmode': 'xml'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        id_list = root.find('IdList')
        
        if id_list is not None and len(id_list) > 0:
            return id_list[0].text
        else:
            print(f"Warning: No taxonomy ID found for '{scientific_name}'")
            return None
            
    except Exception as e:
        print(f"Error getting taxonomy ID for '{scientific_name}': {e}")
        return None

def get_taxonomy_lineage(taxid):
    """Get full taxonomic lineage for a taxonomy ID."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'taxonomy',
        'id': taxid,
        'retmode': 'xml'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        taxon = root.find('Taxon')
        
        if taxon is None:
            return {}
        
        # Get scientific name
        sci_name = taxon.find('ScientificName')
        scientific_name = sci_name.text if sci_name is not None else ""
        
        # Get rank
        rank_elem = taxon.find('Rank')
        rank = rank_elem.text if rank_elem is not None else ""
        
        # Initialize taxonomy dict
        taxonomy = {
            'taxid': taxid,
            'scientific_name': scientific_name,
            'kingdom': '',
            'phylum': '',
            'class': '',
            'order': '',
            'family': '',
            'genus': ''
        }
        
        # Parse lineage
        lineage = taxon.find('LineageEx')
        if lineage is not None:
            for taxon_elem in lineage.findall('Taxon'):
                rank_elem = taxon_elem.find('Rank')
                name_elem = taxon_elem.find('ScientificName')
                
                if rank_elem is not None and name_elem is not None:
                    rank_name = rank_elem.text
                    sci_name = name_elem.text
                    
                    # Map common ranks to our taxonomy dict
                    if rank_name in taxonomy:
                        taxonomy[rank_name] = sci_name
                    elif rank_name == 'superkingdom':
                        # Map superkingdom to kingdom if kingdom is empty
                        if not taxonomy['kingdom']:
                            taxonomy['kingdom'] = sci_name
        
        return taxonomy
        
    except Exception as e:
        print(f"Error getting taxonomy lineage for taxid '{taxid}': {e}")
        return {}

def extract_species_data(html_content):
    """Extract species data from the HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    species_data = []
    
    # Find all fasta download links to get species codes
    fasta_links = soup.find_all('a', href=re.compile(r'/fasta/[a-z]{3}\?'))
    
    processed_codes = set()  # Track processed codes to avoid duplicates
    
    for link in fasta_links:
        href = link.get('href')
        # Extract 3-letter code from URL like /fasta/hsa?pre=1
        match = re.search(r'/fasta/([a-z]{3})\?', href)
        if not match:
            continue
            
        code = match.group(1)
        
        # Skip if we already processed this code
        if code in processed_codes:
            continue
        processed_codes.add(code)
        
        # Find the species name in the same table row
        # Look for the previous table cell that contains the species name
        row = link.find_parent('tr')
        if not row:
            continue
            
        # Find the first cell that contains text (species name)
        cells = row.find_all(['td', 'th'])
        species_name_text = None
        
        for cell in cells:
            text = cell.get_text(strip=True)
            # Skip cells that are just links or empty
            if text and not text.startswith('[') and not text.startswith('Download'):
                species_name_text = text
                break
        
        if not species_name_text:
            print(f"Warning: Could not find species name for code {code}")
            continue
        
        # Parse the species name
        common_name, scientific_name = parse_species_name(species_name_text)
        
        species_data.append({
            'code': code,
            'common_name': common_name,
            'scientific_name': scientific_name
        })
    
    return species_data

def enrich_with_taxonomy(species_data):
    """Add NCBI taxonomy information to species data."""
    print("\nEnriching with NCBI taxonomy data...")
    enriched_data = []
    
    total = len(species_data)
    for i, species in enumerate(species_data, 1):
        print(f"Processing {i}/{total}: {species['scientific_name']}")
        
        # Get taxonomy ID
        taxid = get_ncbi_taxid(species['scientific_name'])
        
        if taxid:
            # Get full lineage
            taxonomy = get_taxonomy_lineage(taxid)
            
            # Merge species data with taxonomy
            enriched_species = {**species, **taxonomy}
        else:
            # If no taxonomy found, add empty fields
            enriched_species = {
                **species,
                'taxid': '',
                'kingdom': '',
                'phylum': '',
                'class': '',
                'order': '',
                'family': '',
                'genus': ''
            }
        
        enriched_data.append(enriched_species)
        
        # Rate limiting: NCBI recommends max 3 requests/second
        # We're making 2 requests per species, so wait 1 second between species
        if i < total:  # Don't wait after the last one
            time.sleep(1)
    
    return enriched_data
    """Extract species data from the HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    species_data = []
    
    # Find all fasta download links to get species codes
    fasta_links = soup.find_all('a', href=re.compile(r'/fasta/[a-z]{3}\?'))
    
    processed_codes = set()  # Track processed codes to avoid duplicates
    
    for link in fasta_links:
        href = link.get('href')
        # Extract 3-letter code from URL like /fasta/hsa?pre=1
        match = re.search(r'/fasta/([a-z]{3})\?', href)
        if not match:
            continue
            
        code = match.group(1)
        
        # Skip if we already processed this code
        if code in processed_codes:
            continue
        processed_codes.add(code)
        
        # Find the species name in the same table row
        # Look for the previous table cell that contains the species name
        row = link.find_parent('tr')
        if not row:
            continue
            
        # Find the first cell that contains text (species name)
        cells = row.find_all(['td', 'th'])
        species_name_text = None
        
        for cell in cells:
            text = cell.get_text(strip=True)
            # Skip cells that are just links or empty
            if text and not text.startswith('[') and not text.startswith('Download'):
                species_name_text = text
                break
        
        if not species_name_text:
            print(f"Warning: Could not find species name for code {code}")
            continue
        
        # Parse the species name
        common_name, scientific_name = parse_species_name(species_name_text)
        
        species_data.append({
            'code': code,
            'common_name': common_name,
            'scientific_name': scientific_name
        })
    
    return species_data

def save_to_csv(species_data, filename='mirgenedb_species_with_taxonomy.csv'):
    """Save species data with taxonomy to CSV file."""
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['3_letter_code', 'common_name', 'taxid']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for species in sorted(species_data, key=lambda x: x['code']):
                writer.writerow({
                    '3_letter_code': species['code'],
                    'common_name': species['common_name'],
                    'taxid': species.get('taxid', '')
                })
        
        print(f"Successfully saved {len(species_data)} species to {filename}")
        
    except IOError as e:
        print(f"Error saving CSV file: {e}")
        sys.exit(1)

def main():
    """Main function to run the extraction process."""
    print("MirGeneDB Species Data Extractor with NCBI Taxonomy")
    print("=" * 55)
    
    # Fetch the page
    html_content = fetch_download_page()
    
    # Extract species data
    print("Parsing species data...")
    species_data = extract_species_data(html_content)
    
    if not species_data:
        print("Error: No species data found!")
        sys.exit(1)
    
    print(f"Found {len(species_data)} species")
    
    # Show first few examples
    print("\nFirst 5 species found:")
    for species in sorted(species_data, key=lambda x: x['code'])[:5]:
        print(f"  {species['code']}: {species['common_name']} ({species['scientific_name']})")
    
    # Ask user if they want to continue with taxonomy enrichment
    print(f"\nThis will make ~{len(species_data) * 2} API calls to NCBI (rate-limited).")
    print(f"Estimated time: ~{len(species_data)} seconds")
    
    response = input("\nProceed with taxonomy enrichment? (y/n): ").lower().strip()
    if response != 'y':
        print("Saving basic species data without taxonomy...")
        # Save basic version
        basic_filename = 'mirgenedb_species_basic.csv'
        with open(basic_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['3_letter_code', 'common_name', 'scientific_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for species in sorted(species_data, key=lambda x: x['code']):
                writer.writerow({
                    '3_letter_code': species['code'],
                    'common_name': species['common_name'],
                    'scientific_name': species['scientific_name']
                })
        print(f"Basic data saved to {basic_filename}")
        return
    
    # Enrich with taxonomy data
    enriched_data = enrich_with_taxonomy(species_data)
    
    # Save enriched data to CSV
    save_to_csv(enriched_data)
    
    # Show summary statistics
    print(f"\nSummary:")
    with_taxid = sum(1 for s in enriched_data if s.get('taxid'))
    print(f"  Species with taxonomy IDs: {with_taxid}/{len(enriched_data)}")
    
    if with_taxid > 0:
        print(f"\nExample enriched entry:")
        example = next((s for s in enriched_data if s.get('taxid')), None)
        if example:
            print(f"  Code: {example['code']}")
            print(f"  Name: {example['common_name']} ({example.get('scientific_name', '')})")
            print(f"  Taxid: {example.get('taxid', '')}")
            print(f"  Kingdom: {example.get('kingdom', '')}")
            print(f"  Phylum: {example.get('phylum', '')}")
            print(f"  Class: {example.get('class', '')}")
            print(f"  Order: {example.get('order', '')}")
            print(f"  Family: {example.get('family', '')}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()