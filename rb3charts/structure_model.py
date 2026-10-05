"""Small auditable baseline for lane-independent musical structure."""


def build_structure_baseline(input_features=130, hidden_features=64):
    return build_structure_model(
        "conv1d",
        input_features=input_features,
        hidden_features=hidden_features,
    )


def build_structure_model(
    architecture,
    input_features=130,
    hidden_features=64,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class StructureModel(nn.Module):
        def __init__(self):
            super().__init__()
            if architecture == "conv1d":
                self.input_projection = nn.Identity()
                self.encoder = nn.Sequential(
                    nn.Conv1d(input_features, hidden_features, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(hidden_features, hidden_features, kernel_size=5, padding=2),
                    nn.ReLU(),
                )
            elif architecture == "dilated-tcn":
                self.input_projection = nn.Conv1d(
                    input_features,
                    hidden_features,
                    kernel_size=1,
                )
                self.encoder = nn.ModuleList([
                    nn.Conv1d(
                        hidden_features,
                        hidden_features,
                        kernel_size=5,
                        dilation=dilation,
                        padding=2 * dilation,
                    )
                    for dilation in (1, 2, 4, 8, 16, 32)
                ])
            elif architecture in (
                "multi-scale",
                "multi-scale-local",
                "multi-scale-offset",
                "multi-scale-gated",
            ):
                self.input_projection = nn.Conv1d(
                    input_features,
                    hidden_features,
                    kernel_size=1,
                )
                self.encoder = nn.Sequential(
                    nn.Conv1d(
                        hidden_features,
                        hidden_features,
                        kernel_size=5,
                        padding=2,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        hidden_features,
                        hidden_features,
                        kernel_size=5,
                        padding=2,
                    ),
                    nn.ReLU(),
                )
                self.context_pools = nn.ModuleList([
                    nn.AvgPool1d(
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        count_include_pad=False,
                    )
                    for kernel_size in (33, 129, 257)
                ])
                self.context_fusion = nn.Conv1d(
                    hidden_features * 4,
                    hidden_features,
                    kernel_size=1,
                )
                if architecture == "multi-scale-local":
                    self.local_attack = nn.Conv1d(
                        hidden_features,
                        1,
                        kernel_size=1,
                    )
                if architecture == "multi-scale-offset":
                    self.attack_offset = nn.Sequential(
                        nn.Conv1d(
                            hidden_features * 2,
                            hidden_features,
                            kernel_size=1,
                        ),
                        nn.ReLU(),
                        nn.Conv1d(hidden_features, 1, kernel_size=1),
                        nn.Tanh(),
                    )
                if architecture == "multi-scale-gated":
                    self.local_attack = nn.Conv1d(
                        hidden_features,
                        1,
                        kernel_size=1,
                    )
                    self.gate_fusion = nn.Sequential(
                        nn.Conv1d(
                            hidden_features * 3,
                            hidden_features,
                            kernel_size=1,
                        ),
                        nn.ReLU(),
                    )
                    self.attack_gate = nn.Conv1d(
                        hidden_features,
                        1,
                        kernel_size=1,
                    )
                    self.gate_smoothing = nn.AvgPool1d(
                        kernel_size=17,
                        stride=1,
                        padding=8,
                        count_include_pad=False,
                    )
            else:
                raise ValueError(f"Unsupported structure architecture: {architecture}")
            self.attack = nn.Conv1d(hidden_features, 1, kernel_size=1)
            self.polyphony = nn.Conv1d(hidden_features, 7, kernel_size=1)
            self.sustain = nn.Conv1d(hidden_features, 7, kernel_size=1)
            self.activation = nn.ReLU()

        def forward(self, features):
            encoded = self.input_projection(features.transpose(1, 2))
            if architecture == "dilated-tcn":
                for layer in self.encoder:
                    encoded = self.activation(encoded + layer(encoded))
            elif architecture in (
                "multi-scale",
                "multi-scale-local",
                "multi-scale-offset",
                "multi-scale-gated",
            ):
                local_encoded = self.encoder(encoded)
                pooled_context = [
                    pool(local_encoded) for pool in self.context_pools
                ]
                encoded = self.activation(self.context_fusion(
                    torch.cat(
                        [local_encoded] + pooled_context,
                        dim=1,
                    )
                ))
            else:
                encoded = self.encoder(encoded)
            outputs = {
                "attack": (
                    self.attack(encoded)
                    + (
                        self.local_attack(local_encoded)
                        if architecture == "multi-scale-local"
                        else 0
                    )
                ).squeeze(1),
                "polyphony": self.polyphony(encoded).transpose(1, 2),
                "sustain": self.sustain(encoded).transpose(1, 2),
            }
            if architecture == "multi-scale-offset":
                outputs["attack_offset"] = self.attack_offset(
                    torch.cat((local_encoded, encoded), dim=1)
                ).squeeze(1)
            if architecture == "multi-scale-gated":
                local_log_probability = functional.logsigmoid(
                    self.local_attack(local_encoded).squeeze(1)
                )
                gate_log_probability = functional.logsigmoid(
                    self.gate_smoothing(self.attack_gate(
                        self.gate_fusion(torch.cat(
                            pooled_context,
                            dim=1,
                        ))
                    )).squeeze(1)
                )
                log_probability = (
                    local_log_probability + gate_log_probability
                )
                probability = torch.exp(log_probability).clamp_max(
                    1 - 1e-6
                )
                outputs["attack"] = (
                    log_probability - torch.log1p(-probability)
                )
            return outputs

    return StructureModel()
