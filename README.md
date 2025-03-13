Spoof Detection Project
Welcome to the Spoof Detection Project! This project is designed to register and recognize images using a simple API setup. Follow the steps below to set up and run the code on your local machine.
Prerequisites
Before you begin, ensure you have the following installed:
  •	Python (version 3.12 or higher)
  •	Visual Studio Code (or any preferred code editor)
  •	Postman for API testing
Run Steps
Step 1: Open the Code
  1.	Launch Visual Studio Code (VS Code).
  2.	Open the main.py file located in the project directory.
Step 2: Configure IP Address and Port
  1.	Scroll to the bottom of the main.py file.
  2.	Locate the section where the IP address and port number are defined.
  3.	Update these values based on your PC's configuration: 
      o	Default Example: 192.168.0.160 (IP) and 1200 (Port).
      o	How to Find Your IP Address: 
      	Open the Command Prompt (CMD) on your PC.
      	Type ipconfig and press Enter.
      	Look for the IPv4 Address under your active network connection (e.g., 192.168.x.x).
  4.	Save the changes to main.py.
Step 3: Set Up Postman for Registration
  1.	Open Postman.
  2.	Set the request method to POST.
  3.	Enter the following URL: 
      http://192.168.0.160:1200/register
      (Replace 192.168.0.160 and 1200 with your configured IP and port if different.)
Step 4: Configure the Request Body
  1.	In Postman, go to the Body tab.
  2.	Select the form-data option.
  3.	Add the following key-value pairs: 
        Key	         Value	      Notes
    collectionId	   DEMO1	      Collection identifier
     threshold        0.5	        Recognition threshold
       image	     (Select File)	  Upload an image file
       name	       (Type a name)	  Name for the image
  4.	Double-check your entries against the provided screenshot (if available).
Step 5: Register the Image
  1.	Click the Send button in Postman.
  2.	The code will execute, and the image will be registered in the system.
  3.	Check the response in Postman for confirmation.
Step 6: Test Image Recognition
  1.	In Postman, update the URL to: 
    http://192.168.0.160:1200/recognize
    (Ensure the IP and port match your configuration.)
  2.	Go to the Body tab and keep the form-data settings.
  3.	Set the image key and upload a new image file for recognition (leave other fields as optional).
  4.	Click the Send button.
  5.	The response will display the recognition result based on the accuracy threshold.
Troubleshooting
  •	Port Conflict: If the port is in use, try a different port number (e.g., 1201) and update it in both main.py and Postman.
  •	IP Issues: Ensure your IP address matches your local network configuration.
  •	No Response: Verify that main.py is running in VS Code before sending requests in Postman.
Notes
  •	The current setup assumes a local server running on your machine.
  •	Adjust the threshold value (e.g., 0.5) to fine-tune recognition accuracy as needed.
Contributing
Feel free to fork this project, submit issues, or contribute enhancements via pull requests!
