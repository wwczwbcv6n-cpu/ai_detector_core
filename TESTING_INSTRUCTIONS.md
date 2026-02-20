# End-to-End Testing Instructions

All the code for the mobile app and the server is now in place. Here is a guide to help you run the entire system for testing.

The system has two main parts:
1.  **The Backend Server:** A Python application that runs the AI model.
2.  **The Mobile App:** The iOS or Android app that you'll use to share images.

**You must have both parts running to test the full functionality.**

---

## 1. Backend Server Setup

The backend is a Python Flask server that listens for image analysis requests from the mobile app.

### Step 1: Install Dependencies

The server requires the `Flask` library in addition to the other packages in `requirements.txt`.

```sh
# Install Flask
pip install Flask

# Ensure all other dependencies are installed
pip install -r requirements.txt
```

### Step 2: Run the Server

Run the `server.py` script from your terminal.

```sh
python3 server.py
```

If successful, you will see output indicating the server has started and the detector is initialized:

```
Detector initialized successfully.
Starting Flask server...
Endpoint available at http://0.0.0.0:8080/analyze
```

The server is now running and ready to accept requests. **Keep this terminal window open.**

---

## 2. Mobile App Setup & Testing

### Step 1: Critical Networking Configuration

For the mobile app to communicate with your server, they must be on the **same Wi-Fi network**.

1.  **Find your computer's local IP address.**
    *   **On macOS:** Go to System Settings > Wi-Fi > Details... > IP Address. It will look like `192.168.1.XX`.
    *   **On Linux:** Run the command `ip addr show` in your terminal and look for your `wlan` or `eth` interface's `inet` address.

2.  **Update the IP address in the mobile app code.**
    *   Open the file: `AIDetectorApp/shared/src/commonMain/kotlin/com/myapplication/common/data/AIDetectorApi.kt`.
    *   Find this line:
        ```kotlin
        val response: AnalysisResult = httpClient.post("http://10.0.2.2:8080/analyze") {
        ```
    *   **Replace `10.0.2.2` with your computer's actual local IP address.** For example:
        ```kotlin
        val response: AnalysisResult = httpClient.post("http://192.168.1.15:8080/analyze") {
        ```
    *   **Note:** The address `10.0.2.2` is a special alias that only works for the official Android Emulator to connect to the host computer. For real devices (both Android and iOS) and for the iOS simulator, you **must** use your computer's network IP.

### Step 2: Run the Android App

1.  Open the `AIDetectorApp` project in Android Studio.
2.  Let Gradle sync.
3.  Run the app on a connected Android device or emulator.
4.  Open any app with images (e.g., Photos, Chrome).
5.  Use the system's "Share" feature on an image.
6.  Select "AI Detector" from the share sheet.
7.  The analysis result should be displayed.

### Step 3: Run the iOS App

1.  First, follow the detailed manual setup instructions located in: `AIDetectorApp/iosApp/ios_share_extension_instructions.md`. **This is a required step.**
2.  After completing the setup in Xcode, run the `ShareExtension` scheme on a connected iOS device or simulator.
3.  Xcode will ask you to choose an app to run; select **Photos**.
4.  In the Photos app, select an image and tap the **Share** button.
5.  Find and select your app's icon in the share sheet.
6.  The analysis result should appear in the view that is presented.

---

This completes the setup. You should now be able to test the full end-to-end flow of sharing an image from your phone and seeing the AI analysis result.
