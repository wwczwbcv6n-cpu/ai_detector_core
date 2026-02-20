# AI Detector API Definition

This document outlines the API endpoint for the AI image detection service.

## Endpoint: `/analyze`

*   **HTTP Method:** `POST`
*   **Description:** Receives an image file, analyzes it using the AI model, and returns the detection result.

### Request

The request must be a `multipart/form-data` POST request with a single field.

*   **Field:** `image`
*   **Content:** The raw image file data (e.g., JPEG, PNG).

#### Example cURL Request:
```bash
curl -X POST -F "image=@/path/to/your/image.jpg" http://your-server-address/analyze
```

### Responses

#### Success Response (`200 OK`)

*   **Content-Type:** `application/json`
*   **Body:** A JSON object containing the probability score and a simple conclusion. The `ai_probability` is a float between 0.0 (definitely real) and 1.0 (definitely AI).

**Example Body (AI-Generated):**
```json
{
  "ai_probability": 0.92,
  "conclusion": "AI-Generated"
}
```

**Example Body (Real):**
```json
{
  "ai_probability": 0.15,
  "conclusion": "REAL"
}
```

---

#### Error Responses

*   **Code:** `400 Bad Request`
*   **When:** The `image` field is missing from the request, or the uploaded file is not a valid/supported image format.
*   **Body:**
    ```json
    {
      "error": "No image file provided or file is invalid."
    }
    ```

*   **Code:** `500 Internal Server Error`
*   **When:** An unexpected error occurs on the server during model loading or image processing.
*   **Body:**
    ```json
    {
      "error": "An internal error occurred during analysis."
    }
    ```
