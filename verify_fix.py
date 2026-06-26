import os
import sys
import json

from backend.ingestion.pipeline import run_pipeline
from backend.core.config import get_config
from backend.ingestion.schemas import ObligationSchema

def read_obligations_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return [ObligationSchema(**ob) for ob in data.get('obligations', [])]

def verify_iccw(obs):
    print("\n--- RBI/2022-23/54 (ICCW ATM) ---")
    print(f"Total Obligations: {len(obs)}")
    check1 = False
    check2 = False
    for ob in obs:
        action_lower = ob.action.lower()
        print(f"  - [{ob.domain}] {ob.actor}: {ob.action}")
        if "iccw transactions" in action_lower and ("without levy of" in action_lower or "without any charge" in action_lower):
            if ob.domain == "Payments":
                check1 = True
        if "withdrawal limits" in action_lower and "iccw transactions" in action_lower and "in-line" in action_lower:
            if ob.domain == "Payments":
                check2 = True
    
    print(f"Check 1 (process ICCW without charges...): {'PASS' if check1 else 'FAIL'}")
    print(f"Check 2 (withdrawal limits in-line...): {'PASS' if check2 else 'FAIL'}")


def verify_nbfc(obs):
    print("\n--- RBI/2025-26/226 (NBFC Capital Adequacy) ---")
    print(f"Total Obligations: {len(obs)}")
    check1 = False
    check2 = False
    for ob in obs:
        action_lower = ob.action.lower()
        print(f"  - [{ob.domain}] {ob.actor}: {ob.action}")
        if "limited review" in action_lower or "audit" in action_lower:
            if ob.domain == "CapitalAdequacy" or ob.domain == "ReportingAudit":
                check1 = True
        if "deduct" in action_lower and ("losses" in action_lower or "owned fund" in action_lower):
            if ob.domain == "CapitalAdequacy" or ob.domain == "Other":
                check2 = True
                
    print(f"Check 1 (limited review/audit...): {'PASS' if check1 else 'FAIL'}")
    print(f"Check 2 (deduct losses...): {'PASS' if check2 else 'FAIL'}")


def verify_calamity(obs):
    print("\n--- RBI/2026-27/46 (Calamity Banking) ---")
    print(f"Total Obligations: {len(obs)}")
    check1 = False
    check2 = False
    for ob in obs:
        action_lower = ob.action.lower()
        print(f"  - [{ob.domain}] {ob.actor} [{ob.severity}]: {ob.action} (Deadline: {ob.deadline.text})")
        if "banking services" in action_lower and "satellite" in action_lower:
            check1 = True
        if "restoration of atm" in action_lower:
            check2 = True
            
    print(f"Check 1 (satellite offices...): {'PASS' if check1 else 'FAIL'}")
    print(f"Check 2 (restoration of ATM...): {'PASS' if check2 else 'FAIL'}")


def main():
    
    # 1. ICCW ATM
    file1 = "data/raw/NT5411AA96313B1B48B7A9D9B53DCFF1CBB5.pdf"
    if os.path.exists(file1):
        res1 = run_pipeline(file1, "auto")
        obs1 = read_obligations_from_json(res1.structured_json_path)
        verify_iccw(obs1)
    else:
        print(f"Not found: {file1}")
        
    # 2. NBFC Capital
    file2 = "data/raw/NT226EBFC6CBC4E134340A707A15512F0196B.pdf"
    if os.path.exists(file2):
        res2 = run_pipeline(file2, "auto")
        obs2 = read_obligations_from_json(res2.structured_json_path)
        verify_nbfc(obs2)
    else:
        print(f"Not found: {file2}")
        
    # 3. Calamity Banking
    file3 = "data/raw/NT46BB3AE9CC38644D9A960F2017BA96D485.pdf"
    if os.path.exists(file3):
        res3 = run_pipeline(file3, "auto")
        obs3 = read_obligations_from_json(res3.structured_json_path)
        verify_calamity(obs3)
    else:
        print(f"Not found: {file3}")


if __name__ == "__main__":
    main()
