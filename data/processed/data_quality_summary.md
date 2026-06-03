# Data Quality Summary

- Fund master schemes: 40
- Fund master duplicate AMFI codes: 0
- AMFI codes missing from NAV history: None
- Source NAV rows after duplicate removal: 46,000
- Observed NAV rows retained: 46,000
- Full forward-filled NAV rows: 64,320
- Investor transaction types: ['Lumpsum', 'Redemption', 'SIP']
- KYC statuses: ['Pending', 'Verified']
- Scheme performance expense ratio range: 0.55% to 1.64%

## Validation Issues
- No blocking validation issues found.

NAV was sorted by AMFI code/date, duplicate dates per fund were removed, and missing calendar days were forward-filled for weekends/holidays.