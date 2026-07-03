import opencc
import os

try:
    from jiayan import CRFPunctuator
except ImportError:
    print("[Error] 'jiayan' library is not installed. Please install it using: pip install jiayan")
    exit(1)

class ClassicalChinesePunctuator:
    def __init__(self, model_path="jiayan_models/punctuator.crfsuite", zip_path="jiayan_models.zip"):
        """
        Initialize the NLP pipeline for punctuating Classical Chinese.
        """
        # Step 1: Initialize OpenCC for Traditional to Simplified
        try:
            self.t2s_converter = opencc.OpenCC('t2s')
        except Exception as e:
            print(f"[Error] Failed to initialize OpenCC (t2s): {e}")
            print("Make sure 'opencc' is installed: pip install opencc-python-reimplemented")
            raise e
            
        # Step 3: Initialize OpenCC for Simplified to Traditional
        try:
            self.s2t_converter = opencc.OpenCC('s2t')
        except Exception as e:
            print(f"[Error] Failed to initialize OpenCC (s2t): {e}")
            raise e
            
        # Extract ZIP if necessary
        if not os.path.exists(model_path) and os.path.exists(zip_path):
            print(f"[Info] Found '{zip_path}'. Extracting models...")
            import zipfile
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(".")
                print(f"[Info] Successfully extracted '{zip_path}'.")
            except Exception as e:
                print(f"[Error] Failed to extract '{zip_path}': {e}")
                
        # Step 2: Initialize Jiayan CRFPunctuator
        lm_path = "jiayan_models/jiayan.klm"
        cut_model_path = "jiayan_models/cut_model"
        punc_model_path = "jiayan_models/punc_model"
        
        if not (os.path.exists(lm_path) and os.path.exists(cut_model_path) and os.path.exists(punc_model_path)):
            print(f"[Warning] Jiayan models not found in 'jiayan_models' directory.")
            print("Please ensure you have extracted 'jiayan_models.zip' correctly.")
            self.is_model_loaded = False
        else:
            try:
                from jiayan import load_lm
                lm = load_lm(lm_path)
                self.punctuator = CRFPunctuator(lm, cut_model_path)
                self.punctuator.load(punc_model_path)
                self.is_model_loaded = True
            except Exception as e:
                print(f"[Error] Failed to load Jiayan model: {e}")
                self.is_model_loaded = False
                
    def process(self, raw_text: str) -> str:
        """
        Runs the end-to-end punctuation pipeline.
        """
        if not raw_text:
            return ""
            
        # 1. Text Preprocessing (Traditional to Simplified)
        simplified_text = self.t2s_converter.convert(raw_text)
        
        # 2. Classical Chinese Punctuation (Jiayan)
        if self.is_model_loaded:
            punctuated_simplified = self.punctuator.punctuate(simplified_text)
        else:
            # Fallback if model isn't loaded: just return the converted text (unpunctuated)
            print("[Warning] Punctuation skipped because model is not loaded.")
            punctuated_simplified = simplified_text
            
        if isinstance(punctuated_simplified, list):
            # Jiayan might return a list of strings depending on the version
            punctuated_simplified = "".join(punctuated_simplified)
            
        # 3. Post-processing (Simplified to Traditional)
        final_traditional_text = self.s2t_converter.convert(punctuated_simplified)
        
        return final_traditional_text

if __name__ == "__main__":
    # Sample unpunctuated Traditional Chinese string
    sample_text = "承府丁五萬八千五百四十人田土十二萬六千一百五十畝稅粟九萬六千三百五十七斛錢十一萬一千八百八十三緡銀一千四百七十七兩"
    
    print("="*60)
    print("  NLP CLASSICAL CHINESE PUNCTUATION PIPELINE")
    print("="*60)
    
    try:
        pipeline = ClassicalChinesePunctuator()
        
        print(f"\n[Raw Input]\n{sample_text}\n")
        
        if not pipeline.is_model_loaded:
            print("[Instructions] Please ensure you have 'jiayan_models.zip' in this directory.")
        else:
            result = pipeline.process(sample_text)
            print(f"[Punctuated Output]\n{result}\n")
            
    except Exception as e:
        print(f"\n[Pipeline Error] {e}")
