from AudioGenderRecognition.main import AudioClassifier

aud = AudioClassifier()
print(aud.find_voice_gender('rec2.wav'))
