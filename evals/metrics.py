import editdistance as ed

def calculate_metrics(predicted_text, transcript):
    if len(predicted_text) == 0 and len(transcript) == 0:
        return 0.0, 0.0  # Both strings empty, perfect match
    
    cer = ed.eval(predicted_text, transcript) / max(len(predicted_text), len(transcript))
    
    pred_spl = predicted_text.split()
    transcript_spl = transcript.split()
    
    if len(pred_spl) == 0 and len(transcript_spl) == 0:
        return cer, 0.0  # Both word lists empty, perfect match
    
    wer = ed.eval(pred_spl, transcript_spl) / max(len(pred_spl), len(transcript_spl))
    return cer, wer

def rem(s):
    return s.replace("\n",'')