#!/usr/bin/env python3
"""
Test script to verify the explain endpoint is working correctly.
"""
import requests
import json

# Test with aspirin to see clean suggestions
url = "http://localhost:8000/explain"
data = {
    "target_id": "P24941",
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Explain endpoint working!")
        print(f"✓ Molecular properties: {result.get('molecular_properties', {})}")
        print(f"✓ Binding analysis: {result.get('binding_analysis', {})}")
        print(f"✓ Structural alerts: {result.get('structural_alerts', [])}")
        print(f"✓ Optimization suggestions: {len(result.get('optimization_suggestions', []))} suggestions")
        print(f"✓ Chemical novelty analysis: {len(result.get('chemical_novelty_analysis', []))} insights")
        print(f"✓ Confidence score: {result.get('confidence_score', 'N/A')}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"✗ Connection error: {e}")
