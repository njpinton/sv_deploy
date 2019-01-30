# Speaker Verification Deployment 

Install packages:

`$ pip install -r requirements.txt`

It has two main functions:
- enroll voice model of speaker
- verify the speaker after enrollment

To enroll user:

`$ python voice_capture_enrollment.py`

It will ask for a name, input name.
It will produce a prompt to ready the user for speaking.
After pressing enter, it will record for 5 seconds.

If speaker is already enrolled, you can choose to overwrite old data or record a new one.

Wait until enrollment is finished.


To verify user:

`$ python voice_capture_verification.py`

Input name of speaker.
Press enter after the prompt.
Record voice for 5 seconds.
Wait for verification.


Enrollment of user will save the speaker's:
- voice in the enroll_voice folder
- numpy array representation in enroll_npy folder
- model representation in MODEL folder

A csv file is generated where all names of speaker and data/time of enrollment is recorded.

Same with verification.
