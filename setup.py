"""py2app build configuration for Overheard.app.

Note: pyproject.toml must NOT be present in the working directory when this
runs — setuptools auto-reads it and sets install_requires, which py2app 0.28+
forbids. build_app.sh handles this by temporarily renaming pyproject.toml.
"""

from setuptools import setup

APP = ["src/overheard/app.py"]

DATA_FILES = []

OPTIONS = {
    "semi_standalone": True,
    "argv_emulation": False,
    "plist": {
        "CFBundleName": "Overheard",
        "CFBundleDisplayName": "Overheard",
        "CFBundleIdentifier": "au.com.overmoro.overheard",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        # Menu bar app — no dock icon
        "LSUIElement": True,
        # Microphone permission
        "NSMicrophoneUsageDescription": (
            "Overheard needs microphone access to record meetings."
        ),
        # Notification style
        "NSUserNotificationAlertStyle": "alert",
    },
    "packages": [
        "overheard",
        "rumps",
        "objc",
        "AppKit",
        "Foundation",
        "UserNotifications",
    ],
    "excludes": [
        # Large ML packages live in system site-packages; not bundled.
        # They are resolved at runtime via semi_standalone mode.
        "torch",
        "torchaudio",
        "whisperx",
        "pyannote",
        "transformers",
        "huggingface_hub",
    ],
}

setup(
    name="Overheard",
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
