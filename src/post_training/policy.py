import string

import torch
from torch import nn

ALPHABET = ["\n", *string.ascii_lowercase, *string.digits, *" .,+-*=_{}[]\":/#"]
ALPHABET += ["<bos>", "<eos>"]


class Tokenizer:
    def __init__(self) -> None:
        self.tokens = {token: index for index, token in enumerate(ALPHABET)}
        self.bos = self.tokens["<bos>"]
        self.eos = self.tokens["<eos>"]

    def encode(self, text: str) -> list[int]:
        unknown = [character for character in text if character not in self.tokens]
        if unknown:
            raise ValueError(f"characters absent from tiny tokenizer: {sorted(set(unknown))}")
        return [self.tokens[character] for character in text]

    def decode(self, ids: list[int]) -> str:
        characters = []
        for index in ids:
            if index == self.eos:
                break
            token = ALPHABET[index]
            if token != "<bos>":
                characters.append(token)
        return "".join(characters)


class TinyPolicy(nn.Module):
    """A recurrent token policy conditioned on a deterministic hashed prompt."""

    def __init__(self, hidden_size: int = 32, max_length: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.prompt_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.token_embedding = nn.Embedding(len(ALPHABET), hidden_size)
        self.position_embedding = nn.Embedding(max_length + 1, hidden_size)
        self.cell = nn.GRUCell(2 * hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, len(ALPHABET))

    def prompt_features(self, prompt: str) -> torch.Tensor:
        features = torch.zeros(self.hidden_size)
        encoded = prompt.encode()
        for index, byte in enumerate(encoded):
            features[(byte + 31 * index) % self.hidden_size] += 1.0
        return features / max(1, len(encoded))

    def initial_state(self, prompt: str) -> torch.Tensor:
        return self.prompt_encoder(self.prompt_features(prompt))

    def step(
        self, state: torch.Tensor, previous: torch.Tensor, position: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position > self.max_length:
            raise ValueError("sequence exceeds configured policy length")
        inputs = torch.cat(
            (self.token_embedding(previous), self.position_embedding.weight[position])
        )
        state = self.cell(inputs, state)
        return self.head(state), state

    def rollout(
        self,
        prompt: str,
        tokenizer: Tokenizer,
        max_length: int,
        generator: torch.Generator,
    ) -> tuple[list[int], torch.Tensor]:
        state = self.initial_state(prompt)
        previous = torch.tensor(tokenizer.bos)
        ids, log_probs = [], []
        for position in range(max_length):
            logits, state = self.step(state, previous, position)
            token_log_probs = torch.log_softmax(logits, dim=-1)
            action = int(torch.multinomial(token_log_probs.exp(), 1, generator=generator))
            log_probs.append(token_log_probs[action])
            if action == tokenizer.eos:
                return ids, torch.stack(log_probs)
            ids.append(action)
            previous = torch.tensor(action)
        logits, _ = self.step(state, previous, len(ids))
        log_probs.append(torch.log_softmax(logits, dim=-1)[tokenizer.eos])
        return ids, torch.stack(log_probs)

    @torch.no_grad()
    def greedy(self, prompt: str, tokenizer: Tokenizer, max_length: int) -> str:
        state = self.initial_state(prompt)
        previous = torch.tensor(tokenizer.bos)
        ids = []
        for position in range(max_length):
            logits, state = self.step(state, previous, position)
            action = int(logits.argmax())
            if action == tokenizer.eos:
                break
            ids.append(action)
            previous = torch.tensor(action)
        return tokenizer.decode(ids)

    def evaluate(
        self, prompt: str, ids: list[int], tokenizer: Tokenizer
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.initial_state(prompt)
        previous = torch.tensor(tokenizer.bos)
        log_probs, entropies, all_log_probs = [], [], []
        for position, action in enumerate([*ids, tokenizer.eos]):
            logits, state = self.step(state, previous, position)
            token_log_probs = torch.log_softmax(logits, dim=-1)
            log_probs.append(token_log_probs[action])
            probabilities = token_log_probs.exp()
            entropies.append(-(probabilities * token_log_probs).sum())
            all_log_probs.append(token_log_probs)
            previous = torch.tensor(action)
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(all_log_probs)

    def response_loss(self, prompt: str, response: str, tokenizer: Tokenizer) -> torch.Tensor:
        ids = tokenizer.encode(response)
        log_probs, _, _ = self.evaluate(prompt, ids, tokenizer)
        return -log_probs.mean()
