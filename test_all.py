import json
import requests
import sys

def run_tests():
    url = "https://bup-hackathon-final.vercel.app/optimize-energy"
    
    print(f"Loading test cases...")
    try:
        with open("BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json not found.")
        return

    cases = data.get("cases", [])
    print(f"Found {len(cases)} cases. Starting tests against {url}...\n")

    passed_count = 0

    for case in cases:
        case_id = case["id"]
        input_payload = case["input"]
        expected_cost = case["expected_output"]["total_cost_bdt"]
        expected_peak = case["expected_output"]["peak_grid_kwh"]
        
        print(f"Testing {case_id}...")
        
        try:
            response = requests.post(url, json=input_payload, timeout=30)
            if response.status_code != 200:
                print(f"  ❌ FAILED: HTTP {response.status_code}")
                print(f"     Response: {response.text}")
                continue
                
            result = response.json()
            actual_cost = result.get("total_cost_bdt")
            actual_peak = result.get("peak_grid_kwh")
            
            # Allow minor floating point tolerance
            cost_match = abs(actual_cost - expected_cost) < 0.01
            peak_match = abs(actual_peak - expected_peak) < 0.01
            
            if cost_match and peak_match:
                print(f"  ✅ PASSED (Cost: {actual_cost:.2f}, Peak: {actual_peak:.2f})")
                passed_count += 1
            else:
                print(f"  ❌ MISMATCH")
                print(f"     Expected: Cost = {expected_cost}, Peak = {expected_peak}")
                print(f"     Actual:   Cost = {actual_cost}, Peak = {actual_peak}")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

    print(f"\n--- Summary ---")
    print(f"{passed_count}/{len(cases)} tests passed.")

if __name__ == "__main__":
    run_tests()
