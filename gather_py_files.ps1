<#
.SYNOPSIS
    Gathers all Python files from a root directory (including subfolders)
    and writes them to an output file with headers indicating their relative paths.

.DESCRIPTION
    This script recursively searches for all files with a .py extension in the specified root directory.
    For each file, it writes a header with the file's relative path followed by the file's content into
    a single output file. This preserves the directory structure in the header labels.

.PARAMETER rootDir
    The root directory to search for Python files. Defaults to the current directory (".").

.PARAMETER outputFile
    The file path for the output text file. Defaults to "all_python_files.txt" in the current directory.

.EXAMPLE
    PS C:\Projects\MyApp> .\gather_py_files.ps1 -rootDir "C:\Projects\MyApp" -outputFile "C:\Projects\MyApp\python_files.txt"
#>

param(
    [string]$rootDir = ".",
    [string]$outputFile = "all_python_files.txt"
)

# Resolve the full path of the root directory.
$fullRoot = (Get-Item $rootDir).FullName

# If the output file exists, remove it.
if (Test-Path $outputFile) {
    Remove-Item $outputFile
}

# Recursively get all .py files and append them to the output file.
Get-ChildItem -Path $rootDir -Filter *.py -Recurse | ForEach-Object {
    # Calculate the relative path from the root.
    $relativePath = $_.FullName.Substring($fullRoot.Length + 1)  # +1 to remove the leading separator
    $header = "===== File: $relativePath ====="
    
    # Append header and content to the output file.
    Add-Content -Path $outputFile -Value $header
    Add-Content -Path $outputFile -Value (Get-Content -Path $_.FullName -Raw)
    Add-Content -Path $outputFile -Value "`n"  # New line for clarity between files
}

Write-Output "Python files have been gathered and written to: $outputFile"
