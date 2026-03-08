import chess
import re
import random
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from chess_tournament import Player

class TransformerPlayer(Player):
    # Class variables: shared model across all instances
    _model = None
    _tokenizer = None
    _device = None
    
    def __init__(
        self, 
        name: str = "TransformerPlayer",
        model_id: str = "shawnno/chess-smollm2",  # 添加模型参数
        max_attempts: int = 3,
        **kwargs  # 接收额外参数并传递给父类
    ):
        super().__init__(name, **kwargs)  # 传递所有参数给父类
        self.model_id = model_id
        self.max_attempts = max_attempts
        self.uci_re = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")
        
    def _ensure_model_loaded(self):
        """Lazy loading: load model only when needed"""
        if TransformerPlayer._model is None:
            try:  # 添加异常处理
                print(f"{self.name}: First use, loading model {self.model_id}...")
                TransformerPlayer._device = "cuda" if torch.cuda.is_available() else "cpu"
                
                TransformerPlayer._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                TransformerPlayer._model = AutoModelForCausalLM.from_pretrained(self.model_id)
                TransformerPlayer._model = TransformerPlayer._model.to(TransformerPlayer._device)
                TransformerPlayer._model.eval()
                
                if TransformerPlayer._tokenizer.pad_token is None:
                    TransformerPlayer._tokenizer.pad_token = TransformerPlayer._tokenizer.eos_token
                print(f"{self.name}: Model loaded successfully")
            except Exception as e:
                print(f"{self.name}: Failed to load model - {e}")
                raise  # 重新抛出，让get_move处理
    
    def _random_legal(self, fen: str) -> Optional[str]:
        """Fallback: return random legal move"""
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        return random.choice(legal_moves) if legal_moves else None
    
    def get_move(self, fen: str) -> Optional[str]:
        # Load model on first move request
        try:
            self._ensure_model_loaded()
        except Exception:
            return self._random_legal(fen)  # 模型加载失败时fallback
        
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return None

        for attempt in range(self.max_attempts):
            try:
                prompt = f"<|fen|>{fen}<|move|>"
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=6,
                        do_sample=False,
                        pad_token_id=self._tokenizer.eos_token_id
                    )

                decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if "<|move|>" in decoded:
                    move_part = decoded.split("<|move|>")[-1].strip()
                else:
                    move_part = decoded
                
                match = self.uci_re.search(move_part)
                if match:
                    move = match.group(0)
                    if move in legal_moves:
                        return move
                        
            except Exception:
                continue
        
        return self._random_legal(fen)