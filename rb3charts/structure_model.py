"""Small auditable baseline for lane-independent musical structure."""


def build_structure_baseline(input_features=130, hidden_features=64):
    import torch.nn as nn

    class StructureBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(input_features, hidden_features, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden_features, hidden_features, kernel_size=5, padding=2),
                nn.ReLU(),
            )
            self.attack = nn.Conv1d(hidden_features, 1, kernel_size=1)
            self.polyphony = nn.Conv1d(hidden_features, 7, kernel_size=1)
            self.sustain = nn.Conv1d(hidden_features, 7, kernel_size=1)

        def forward(self, features):
            encoded = self.encoder(features.transpose(1, 2))
            return {
                "attack": self.attack(encoded).squeeze(1),
                "polyphony": self.polyphony(encoded).transpose(1, 2),
                "sustain": self.sustain(encoded).transpose(1, 2),
            }

    return StructureBaseline()
