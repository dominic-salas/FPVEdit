; Inno Setup 6 script: turns the standalone build (dist\FPVEdit, from
; scripts/build_windows.py) into one installer. The release workflow runs it as
;   iscc /DAppVersion=1.0.0 packaging\installer.iss
; Installs per user by default (no admin rights needed), or for all users.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6F0B5C2E-4D7A-4E7B-9A51-2C3E8F1D7A40}
AppName=FPVEdit
AppVersion={#AppVersion}
AppPublisher=Dominic Salas
DefaultDirName={autopf}\FPVEdit
DefaultGroupName=FPVEdit
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=FPVEdit-{#AppVersion}-setup
SetupIconFile=..\build\windows\fpvedit.ico
UninstallDisplayIcon={app}\FPVEdit.exe
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\FPVEdit\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\FPVEdit"; Filename: "{app}\FPVEdit.exe"
Name: "{autodesktop}\FPVEdit"; Filename: "{app}\FPVEdit.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\FPVEdit.exe"; Description: "Start FPVEdit"; Flags: nowait postinstall skipifsilent

[Code]
// The app keeps measurements (to find shots again quickly), settings and a log in the
// user's local app-data folder. Offer to remove them; never touch anything else.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Data: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    Data := ExpandConstant('{localappdata}\FPVEdit');
    if DirExists(Data) and (not UninstallSilent) then
      if MsgBox('Also remove FPVEdit''s saved measurements and log (' + Data + ')? ' +
                'Your videos are not affected.', mbConfirmation, MB_YESNO) = IDYES then
        DelTree(Data, True, True, True);
  end;
end;
