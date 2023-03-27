from google.cloud import speech_v1 as speech
from tqdm import tqdm


def speech_to_text(config, audio):
    client = speech.SpeechClient()
    operation = client.long_running_recognize(config=config, audio=audio)
    print("Waiting for operation to complete...")
    response = operation.result(timeout=6000)
    return response


def get_transcription(response):
    trans_list = []
    for result in response.results:
        best_alternative = result.alternatives[0]
        transcript = best_alternative.transcript
        trans_list.append(transcript)
    return trans_list


def print_sentences(response):
    for result in response.results:
        best_alternative = result.alternatives[0]
        transcript = best_alternative.transcript
        confidence = best_alternative.confidence
        print("-" * 80)
        print(f"Transcript: {transcript}")
        print(f"Confidence: {confidence:.0%}")


def run_trans(file_name, out_name):
    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=10,
    )
    config = dict(language_code="de-DE",
                  enable_automatic_punctuation=True,
                  diarization_config=diarization_config,
                  )
    # audio = {"content": open(file_name, "rb").read()}
    gcs_uri = "gs://speech_text_test_servus/01.wav"
    audio = speech.RecognitionAudio(uri=gcs_uri)
    print("getting response...")
    response = speech_to_text(config=config, audio=audio)
    print("response got")
    trans_list = get_transcription(response)
    with open(out_name, "w+", encoding="utf-8") as out_file:
        for line in tqdm(trans_list):
            out_file.write(line)
            out_file.write("\n")


def main():
    file_name = "data/test_3.wav"
    out_name = "data/01.txt"
    run_trans(file_name, out_name)


if __name__ == '__main__':
    main()
