#!/usr/bin/env python3
"""
Quick test to check the explain API response
"""
import requests
import json

def test_explain_api():
    url = "http://localhost:8000/explain"
    data = {
        "target_id": "P24941",
        "smiles": "COc1ccc(cc1)C(=O)Nc2ccc(cc2)S(=O)(=O)N"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response keys: {list(result.keys())}")
            print(f"Full response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_explain_api()
