# Install LungCAD using SideStore

Download `LungCAD-SideStore-unsigned-device-IPA` from the successful build run. Extract the GitHub ZIP and save `LungCAD-SideStore.ipa` in Files on your iPad. Import that IPA using SideStore's add control. Do not extract the IPA itself.

SideStore signs the app using your Apple Account. Follow https://docs.sidestore.io/docs/installation/install and https://docs.sidestore.io/docs/installation/prerequisites for setup, Developer Mode, trust, Wi-Fi and LocalDevVPN. A paid developer account is not required for ordinary free-account sideloading. Refresh before signing expires, normally seven days for free accounts. LungCAD requires iPadOS 17 or newer.

The model is separate: download `LungCAD-experimental-model-synthetic-parity-only` from a successful model validation run. Extract it into Files and import the folder containing both `lungcad.onnx` and `lungcad.json` in LungCAD. Import CT locally. Never upload patient scans to GitHub. Inference is offline after model import; SideStore signing refresh separately needs a network connection.

CI verifies the arm64 iPhoneOS executable, iPad metadata and IPA archive. Physical iPad installation and performance remain unverified. Synthetic native simulator inference tests passed; clinical accuracy remains unverified. Experimental research use only.
