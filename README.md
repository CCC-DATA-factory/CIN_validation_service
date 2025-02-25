# CIN Validator API

## What Does This Service Do?

Our **CIN Validator API** helps you quickly verify and process images of your Card Identification Numbers (CIN). It uses smart image recognition to detect the card in your picture, calculate useful metrics, and even provide a cropped version of the card. Whether it's the front or the back, our API has got you covered!

---

## How To Set Up the Service

### Locally (Using Python)

**Requirements:** Python 3.11 and a few handy packages.

1. **Clone the repository** and open the project folder.
    ```bash
    git clone https://github.com/CCC-DATA-factory/CIN_validation_service
    cd CIN_validation_service
    ```
2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```
3. **Activate your environment:**

   ```bash
    .\venv\Scripts\activate
   ```
4. **Install the required packages:**
   ```bash
    pip install -r requirements.txt
   ```
5. **Adjust settings:** (like template paths) by editing the .env file.
6. **Run the application:**
   ```bash
    uvicorn cin_validator_api:app --host 0.0.0.0 --port 8082 
    
    #in case the port 8082 is used change it to another unused

   ```
### Using Docker
1. **Clone the repository** and enter the project directory.
     ```bash
    git clone https://github.com/CCC-DATA-factory/CIN_validation_service
    cd CIN_validation_service
    ```
2. **Launch the service with Docker Compose:**
   ```bash
   docker-compose up -d
   ```
## API Endpoints

The service offers several easy-to-use **endpoints**:

## `/validate/front`

**Description:**  
Validates whether the input image contains the front side of the CIN card.

**Method:**  
`POST`

**Input:**  
- **file**: A normal image file of the CIN card (the service converts it to grayscale internally).

**Output:**  
A JSON response with the following schema:

```json
{
  "front_detected": true,
  "score": {
    "good_matches": 123,
    "inliers": 130,
    "inlier_ratio": 1.06
  },
  "metrics": {
    "inferance_time": "0.42 seconds",
    "cpu_usage": "0.00% of 1 core",
    "memory_usage": "0.10 MB"
  }
}
```
Response Fields:

- ``front_detected: Boolean``

    Indicates whether the front side of the CIN card was detected.

- ``score: Object``
Contains detailed metrics:

- ``good_matches (Integer)``: The number of good feature matches found.
- ``inliers (Integer)``: The number of inlier matches determined via homography.
- ``inlier_ratio (Float)``: The ratio of inliers to good matches.
- metrics: Object
Provides performance metrics:

    - ``inferance_time (String)``: Time taken to process the image.
    - ``cpu_usage (String)`` : CPU usage during processing.
    - ``memory_usage (String)``: Memory usage during processing.
## ``/validate/back``
**Description**:
Validates whether the input image contains the back side of the CIN card.

**Method**:
``POST``

**Input:**

- ``file``: A normal image file of the CIN card.

**Output:**
A JSON response with the following schema:

```json
{
  "back_detected": false,
  "score": {
    "good_matches": 98,
    "inliers": 45,
    "inlier_ratio": 0.46
  },
  "metrics": {
    "inferance_time": "0.35 seconds",
    "cpu_usage": "0.00% of 1 core",
    "memory_usage": "0.08 MB"
  }
}
```
## ``/croped/front``
**Description:**
Returns a cropped version of the front side of the CIN card if it is detected in the input image.

**Method:**
``POST``

**Input:**

``file``: A normal image file of the CIN card.

**Output:**
A JPEG image stream containing the cropped front side of the CIN card.

## ``/croped/back``
**Description:**
Returns a cropped version of the back side of the CIN card if it is detected in the input image.

**Method:**
``POST``

**Input:**

- ``file``: A normal image file of the CIN card.

**Output:**
A JPEG image stream containing the cropped back side of the CIN card.


## Environment Variables
You can fine-tune the service using these settings:

- **TEMPLATE_FRONT_PATH**: Location of the front template (e.g., templates/front.jpeg).
- **TEMPLATE_BACK_PATH**: Location of the back template (e.g., templates/back.jpeg).
- **MIN_GOOD_MATCHES**: Minimum required matches (default: 120).
- **INLIER_THRESHOLD_FRONT**: Minimum inlier matches for the front (default: 120).
- **INLIER_THRESHOLD_BACK**: Minimum inlier matches for the back (default: 50).
- **RATIO_THRESHOLD**: Lowe's ratio threshold for filtering (default: 0.7).
These can be set in your Docker Compose file or in a local .env file.

## **Errors & Troubleshooting**
If you run into issues, consider these tips:

- **Invalid image file**: Ensure the uploaded image is a valid grayscale image.
- **Card not detected**: The image may not be clear—check its quality and content.
- **Cropped image error**: Verify the image is valid and try again.
For more details, check the logs and confirm all settings are correctly configured.