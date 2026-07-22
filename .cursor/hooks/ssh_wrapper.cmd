@echo off
REM Bridge for 32-bit msysgit: Sysnative reaches real System32 OpenSSH.
set "SSH="
if exist "%windir%\Sysnative\OpenSSH\ssh.exe" set "SSH=%windir%\Sysnative\OpenSSH\ssh.exe"
if not defined SSH if exist "%windir%\System32\OpenSSH\ssh.exe" set "SSH=%windir%\System32\OpenSSH\ssh.exe"
if not defined SSH if exist "%ProgramFiles(x86)%\Git\bin\ssh.exe" set "SSH=%ProgramFiles(x86)%\Git\bin\ssh.exe"
if not defined SSH (
  echo ssh_wrapper: OpenSSH not found 1>&2
  exit /b 1
)
"%SSH%" -o BatchMode=yes -o StrictHostKeyChecking=accept-new %*
