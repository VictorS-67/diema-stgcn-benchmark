"""Tests for YAML configuration loading."""

import pytest
import yaml

from emo_mocap.tools.config import load_config


@pytest.fixture
def valid_config(tmp_path):
    """Create a minimal valid config YAML and return its path."""
    config = {
        "data": {"data_path": "data/test.pkl"},
        "model": {"type": "stgcn", "num_class": 7},
        "skeleton": {
            "num_nodes": 25,
            "inward_edges": [[0, 1], [2, 1]],
        },
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


class TestLoadConfig:
    """Tests for load_config."""

    def test_parse_valid_config(self, valid_config):
        cfg = load_config(valid_config)
        assert cfg.model.type == "stgcn"
        assert cfg.model.num_class == 7
        assert cfg.skeleton.num_nodes == 25

    def test_defaults_applied(self, valid_config):
        cfg = load_config(valid_config)
        assert cfg.training.base_lr == 0.1
        assert cfg.training.optimizer == "SGD"
        assert cfg.training.max_epochs == 100
        assert cfg.model.in_channels == 3
        assert cfg.model.dropout == 0.5

    def test_precision_default_preserves_legacy_behavior(self, valid_config):
        # Default must stay float32 so existing configs/tests don't silently
        # flip to mixed precision.
        cfg = load_config(valid_config)
        assert cfg.training.precision == "32-true"
        assert cfg.training.devices == "auto"

    def test_precision_user_override(self, tmp_path):
        config = {
            "data": {"data_path": "data/test.pkl"},
            "model": {"type": "stgcn", "num_class": 7},
            "skeleton": {"num_nodes": 25, "inward_edges": [[0, 1]]},
            "training": {"precision": "bf16-mixed", "devices": 1},
        }
        path = tmp_path / "precision.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)
        cfg = load_config(path)
        assert cfg.training.precision == "bf16-mixed"
        assert cfg.training.devices == 1

    def test_user_values_override_defaults(self, tmp_path):
        config = {
            "data": {"data_path": "data/test.pkl"},
            "model": {"type": "stgcn", "num_class": 7, "dropout": 0.3},
            "skeleton": {
                "num_nodes": 25,
                "inward_edges": [[0, 1]],
            },
            "training": {"base_lr": 0.01, "max_epochs": 50},
        }
        path = tmp_path / "override.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        cfg = load_config(path)
        assert cfg.model.dropout == 0.3
        assert cfg.training.base_lr == 0.01
        assert cfg.training.max_epochs == 50
        # Defaults still applied for missing keys
        assert cfg.training.optimizer == "SGD"

    def test_missing_required_section_raises(self, tmp_path):
        config = {
            "model": {"type": "stgcn", "num_class": 7},
            "skeleton": {"num_nodes": 25, "inward_edges": [[0, 1]]},
        }
        path = tmp_path / "no_data.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="missing required section 'data'"):
            load_config(path)

    def test_missing_required_field_raises(self, tmp_path):
        config = {
            "data": {"data_path": "data/test.pkl"},
            "model": {"type": "stgcn"},  # missing num_class
            "skeleton": {"num_nodes": 25, "inward_edges": [[0, 1]]},
        }
        path = tmp_path / "no_class.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="missing required field 'model.num_class'"):
            load_config(path)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_malformed_yaml_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("not: a: valid: yaml: [")
        with pytest.raises(yaml.YAMLError):
            load_config(path)

    def test_non_dict_yaml_raises(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            load_config(path)

    def test_real_config_files(self):
        """Validate the example config files ship correctly."""
        from pathlib import Path
        configs_dir = Path(__file__).parent.parent / "configs"
        for yaml_file in configs_dir.glob("*.yaml"):
            cfg = load_config(yaml_file)
            assert hasattr(cfg, "model")
            assert hasattr(cfg.model, "type")
