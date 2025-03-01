#                       discover-AI
# #############################################################
#               written by Dzebu DS                           #
# #############################################################

an api app, made with flask,html,css and python 
it calls hugging_face facebook , summerisser and paraphrase api
to transcript audio, analyze the resultant text and asign
sentimental score to the audio recording and suggest a product
from offerings


#                          HOW TO RUN
_______________________________________________________________________________________
1. run the python main file: python3 main.py
2. on web browser go to: 127.0.0.1:5000
3: upload your recording and await results
________________________________________________________________________________________
# TODO: refine summarization to just extract keywords that are related to rating products
# TODO: add database intergration for stroring products and transcripts
# TODO: re-intergrate the database into the api read/write calls
# TODO: remove locally stored files once posting(to db) is complete
# TODO: improve front end to better align with current UI'S
