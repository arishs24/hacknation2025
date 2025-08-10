#!/usr/bin/env python3
"""
Test script to verify molecular properties are calculated uniquely for different molecules.
"""
import requests
import json

# Test different molecules
test_molecules = [
    {
        "name": "Aspirin",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "expected_mw": 180.16
    },
    {
        "name": "Caffeine", 
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "expected_mw": 194.19
    },
    {
        "name": "Ibuprofen",
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 
        "expected_mw": 206.28
    },
    {
        "name": "Large molecule",
        "smiles": "COc1ccc(cc1)C(=O)Nc2ccc(cc2)S(=O)(=O)N",
        "expected_mw": 306.34
    }
]

url = "http://localhost:8000/explain"

print("Testing molecular property calculation for different molecules:")
print("=" * 60)

for molecule in test_molecules:
    data = {
        "target_id": "P24941",
        "smiles": molecule["smiles"]
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            props = result.get('molecular_properties', {})
            actual_mw = props.get('molecular_weight', 0)
            
            print(f"\n{molecule['name']}:")
            print(f"  SMILES: {molecule['smiles']}")
            print(f"  Expected MW: {molecule['expected_mw']}")
            print(f"  Actual MW: {actual_mw}")
            print(f"  LogP: {props.get('logp', 'N/A')}")
            print(f"  HBD: {props.get('hbd', 'N/A')}")
            print(f"  HBA: {props.get('hba', 'N/A')}")
            print(f"  TPSA: {props.get('tpsa', 'N/A')}")
            print(f"  Drug-likeness: {props.get('drug_likeness_score', 'N/A')}")
            
            # Check if properties are being calculated correctly
            if abs(actual_mw - molecule['expected_mw']) < 1.0:
                print(f"  ✅ Molecular weight calculation correct!")
            else:
                print(f"  ❌ Molecular weight mismatch!")
        else:
            print(f"❌ API error for {molecule['name']}: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Connection error for {molecule['name']}: {e}")

print("\n" + "=" * 60)
