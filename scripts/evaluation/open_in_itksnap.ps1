# Open a case in ITK-SNAP for quick mask editing.
# Usage:  scripts/evaluation/open_in_itksnap.ps1 <case_id>
# Requires ITK-SNAP installed and either:
#   - ITKSNAP_EXE env var set to the .exe path, or
#   - ITK-SNAP in PATH as 'ITK-SNAP.exe' / 'itksnap.exe'
#
# Loads t1_gd.nii.gz as the primary image and seg.nii.gz as the segmentation.
param(
    [Parameter(Mandatory = $true)]
    [string]$CaseId
)

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# Locate case directory (BMS/UCSF in preprocessed_256, Mets_ in data/train)
$candidates = @(
    (Join-Path $root "data\preprocessed_256\train\$CaseId"),
    (Join-Path $root "data\train\$CaseId")
)
$caseDir = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $caseDir) {
    Write-Error "Case not found: $CaseId"
    exit 1
}

$t1 = Join-Path $caseDir "t1_gd.nii.gz"
$seg = Join-Path $caseDir "seg.nii.gz"
if (-not (Test-Path $t1)) { Write-Error "Missing: $t1"; exit 1 }
if (-not (Test-Path $seg)) { Write-Error "Missing: $seg"; exit 1 }

# Find ITK-SNAP executable
$exe = $env:ITKSNAP_EXE
if (-not $exe) {
    foreach ($name in @("ITK-SNAP.exe", "itksnap.exe")) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if ($found) { $exe = $found.Source; break }
    }
}
if (-not $exe) {
    $common = @(
        "C:\Program Files\ITK-SNAP 4.2\bin\ITK-SNAP.exe",
        "C:\Program Files\ITK-SNAP 4.0\bin\ITK-SNAP.exe",
        "C:\Program Files\ITK-SNAP 3.8\bin\ITK-SNAP.exe"
    )
    $exe = $common | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $exe) {
    Write-Error "ITK-SNAP not found. Set `$env:ITKSNAP_EXE or install ITK-SNAP."
    exit 1
}

Write-Host "Opening $CaseId in ITK-SNAP..."
Write-Host "  T1 post-contrast: $t1"
Write-Host "  Segmentation:     $seg"
Write-Host ""
Write-Host "When done editing, save the mask (e.g. as $CaseId`_edited.nii.gz) and run:"
Write-Host "  python scripts/evaluation/commit_mask_fix.py $CaseId <path-to-edited-mask.nii.gz>"
Write-Host ""

& $exe --main $t1 --segmentation $seg
