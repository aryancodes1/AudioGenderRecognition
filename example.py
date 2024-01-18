from main import AudioClassifier

aud = AudioClassifier()
aud.record_voice("antriksh.wav")
print(aud.find_voice_gender('/Users/arunkaul/Desktop/AudioSignal/antriksh.wav'))#Add File Path

