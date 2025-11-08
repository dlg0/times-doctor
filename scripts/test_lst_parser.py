#!/usr/bin/env python3
"""Test LST parser on actual file."""

import json
from pathlib import Path
from times_doctor.lst_parser import process_lst_file

# Test on actual LST file
lst_file = Path('data/065Nov25-annualupto2045/parscen/parscen~0011/parscen~0011.lst')

if not lst_file.exists():
    print(f"File not found: {lst_file}")
    exit(1)

print(f"Processing: {lst_file}")
print(f"File size: {lst_file.stat().st_size / 1024 / 1024:.2f} MB")
print()

result = process_lst_file(lst_file)

print("=" * 80)
print("METADATA")
print("=" * 80)
print(json.dumps(result['metadata'], indent=2))
print()

print("=" * 80)
print("SECTIONS FOUND")
print("=" * 80)
for section_name in result['sections'].keys():
    print(f"  - {section_name}")
print()

# Show details for each section type
for section_name, section_data in result['sections'].items():
    print("=" * 80)
    print(f"SECTION: {section_name}")
    print("=" * 80)
    
    # Show text summary if available
    if 'text_summary' in section_data:
        print(section_data['text_summary'])
        print()
    
    # For compilation, show error summary
    if 'errors' in section_data:
        print("Error details:")
        for error_code, error_info in section_data['errors'].items():
            print(f"\n  Error {error_code}:")
            print(f"    Total occurrences: {error_info['count']}")
            print(f"    Unique patterns: {len(error_info['elements'])}")
            
            # Show top 3 patterns
            top_patterns = sorted(
                error_info['elements'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            if top_patterns:
                print(f"    Top patterns:")
                for pattern, count in top_patterns:
                    print(f"      - {pattern}: {count}")
    
    # For execution/model analysis, show summary
    if 'summary' in section_data:
        print("\nSummary statistics:")
        if isinstance(section_data['summary'], dict):
            for key, value in section_data['summary'].items():
                print(f"  {key}: {value}")
        else:
            print(f"  {section_data['summary']}")
    
    print()

# Save to JSON for inspection
output_file = Path('lst_parsed_output.json')
with open(output_file, 'w') as f:
    # Remove full content fields to make JSON manageable
    clean_result = result.copy()
    for section_name, section_data in clean_result['sections'].items():
        if 'content' in section_data and len(str(section_data['content'])) > 1000:
            section_data['content'] = section_data['content'][:1000] + "... (truncated)"
    
    json.dump(clean_result, f, indent=2)

print(f"\nFull results saved to: {output_file}")
print(f"Output size: {output_file.stat().st_size / 1024:.2f} KB")
