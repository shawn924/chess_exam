import chess
import re
import random
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from chess_tournament import Player

class TransformerPlayer(Player):
    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained("shawnno/chess-smollm2")
        self.model = AutoModelForCausalLM.from_pretrained("shawnno/chess-smollm2")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.uci_re = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return None

        for attempt in range(3):
            try:
                prompt = f"<|fen|>{fen}<|move|>"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=6,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
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
        
        return random.choice(legal_moves)