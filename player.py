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
        max_attempts: int = 5,
        **kwargs  # 接收额外参数并传递给父类
    ):
        super().__init__(name, **kwargs)  # 传递所有参数给父类
        self.model_id = model_id
        self.max_attempts = max_attempts
        self.uci_re = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")
        
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
        legal_moves = list(board.legal_moves)
        if not legal_moves: return None
    
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()

        capture_moves = [m for m in legal_moves if board.is_capture(m)]
        if capture_moves:
            # 随机从吃子步法里选一个，也比纯随机好得多
            return random.choice(capture_moves).uci()
        
        return random.choice(legal_moves).uci()
    
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
        
        prompt = f"<|fen|>{fen}<|move|>"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        input_len = inputs.input_ids.shape[1]

        for attempt in range(self.max_attempts):
            try:
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=6,     
                        do_sample=True,       
                        temperature=0.4,      
                        top_p=0.9,          
                        pad_token_id=self._tokenizer.eos_token_id
                    )

                new_tokens = outputs[0][input_len:]
                move_part = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                
                match = self.uci_re.search(move_part)
                if match:
                    move = match.group(0)
                    if move in legal_moves:
                        return move
                        
            except Exception:
                continue
        
        return self._random_legal(fen)