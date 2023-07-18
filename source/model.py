# Import the Speech-to-Text client library
# from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel
import openai
import os


# openAI Key 
openai.api_key = os.environ["API_KEY"]

def language_model1(prompt):
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic.schema import predict
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value

    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(
        client_options=client_options
    )
    instance_dict = {
        "context": "",
        "examples": [],
        "messages": [
            {
                "author": "user",
                "content": prompt
            },
            {
                "content": "Hello, how can I help you today?",
                "author": "bot",
                "citationMetadata": {
                    "citations": []
                }
            }
        ]
    }
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {
        "temperature": 0.2,
        "maxOutputTokens": 256,
        "topP": 0.8,
        "topK": 40
    }
    parameters = json_format.ParseDict(parameters_dict, Value())
    response = client.predict(
        endpoint=os.environ["path"], instances=instances, parameters=parameters
    )
    
    predictions = response.predictions
    for prediction in predictions:
        # print(" prediction:", dict(prediction))
        pass
    return predictions

def language_model2(prompt):
    import vertexai
    from vertexai.preview.language_models import ChatModel, InputOutputTextPair

    vertexai.init(project=os.environ["project-id"], location=os.environ["location"])
    chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    chat = chat_model.start_chat()
    response = chat.send_message(prompt, **parameters)
    return response.text


# model for Text to Speech
def model_tts(prompt,voice_out):
    from google.cloud import texttospeech
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=prompt)
    voice = texttospeech.VoiceSelectionParams(
        language_code="es-US",
        name=voice_out,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=1
    )
    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    with open("result/output.mp3", "wb") as out:
        out.write(response.audio_content)

# model for Speech to  Text
def model_stt(audio_file):
    # Import the Speech-to-Text client library
    from google.cloud import speech

    # Instantiates a client
    client = speech.SpeechClient()

    # The content of the audio file to transcribe
    audio_content = audio_file
    # transcribe speech
    audio = speech.RecognitionAudio(content=audio_content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=24000,
        language_code="en-US",
        model="default",
        audio_channel_count=1,
        enable_word_confidence=True,
        enable_word_time_offsets=True,
    )

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=90)
    final=""
    for result in response.results:
        final=final + result.alternatives[0].transcript
    return final

# model for image generation
def model_img(prompt):
  response = openai.Image.create(
    prompt=prompt,
    n=1,
    size="1024x1024"
  )
  image_url = response['data'][0]['url']
  return (image_url)