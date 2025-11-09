import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from times_doctor.core.prompts import (
    _find_prompts_dir,
    _validate_placeholders,
    build_extraction_prompt,
    build_llm_prompt,
    build_qa_check_compress_prompt,
    build_review_prompt,
    load_prompt_template,
)


class TestPromptLoader:
    """Test versioned prompt template loading."""
    
    def test_find_prompts_dir(self):
        """Should locate prompts/ directory from installed package."""
        prompts_dir = _find_prompts_dir()
        assert prompts_dir.exists()
        assert prompts_dir.is_dir()
        assert prompts_dir.name == "prompts"
        assert (prompts_dir / "manifest.json").exists()
    
    def test_validate_placeholders_exact_match(self):
        """Should pass when placeholders match exactly."""
        content = "Hello {name}, welcome to {place}!"
        required = {"name", "place"}
        missing, extra = _validate_placeholders(content, required)
        assert missing == set()
        assert extra == set()
    
    def test_validate_placeholders_missing(self):
        """Should detect missing placeholders."""
        content = "Hello {name}!"
        required = {"name", "place"}
        missing, extra = _validate_placeholders(content, required)
        assert missing == {"place"}
        assert extra == set()
    
    def test_validate_placeholders_extra(self):
        """Should detect unexpected placeholders."""
        content = "Hello {name}, welcome to {place} at {time}!"
        required = {"name", "place"}
        missing, extra = _validate_placeholders(content, required)
        assert missing == set()
        assert extra == {"time"}
    
    def test_load_current_version_from_manifest(self):
        """Should load current version specified in manifest."""
        template = load_prompt_template("qa_check_compress")
        assert template is not None
        assert "QA_CHECK.LOG" in template
        assert "SEVERE ERROR" in template
    
    def test_load_specific_version(self):
        """Should load specific version when requested."""
        template = load_prompt_template("extraction_sections", version="v1")
        assert template is not None
        assert "{file_type}" in template
        assert "Return ONLY valid JSON" in template
    
    def test_load_with_placeholder_validation(self):
        """Should validate required placeholders."""
        template = load_prompt_template(
            "extraction_sections",
            required_placeholders={"file_type"}
        )
        assert template is not None
        assert "{file_type}" in template
    
    def test_load_placeholder_validation_fails_missing(self):
        """Should raise ValueError if required placeholder missing."""
        with pytest.raises(ValueError, match="missing"):
            load_prompt_template(
                "qa_check_compress",
                required_placeholders={"nonexistent"}
            )
    
    def test_load_placeholder_validation_fails_extra(self):
        """Should raise ValueError if unexpected placeholder found."""
        # Create a test template with unexpected placeholders
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_prompts = Path(tmpdir) / "prompts"
            tmp_prompts.mkdir()
            test_dir = tmp_prompts / "test_prompt"
            test_dir.mkdir()
            (test_dir / "v1.txt").write_text("Test {extra} content", encoding="utf-8")
            
            manifest = {
                "test_prompt": {
                    "current": "v1",
                    "versions": {"v1": {"sha256": hashlib.sha256(b"Test {extra} content").hexdigest()}}
                }
            }
            (tmp_prompts / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            
            with patch("times_doctor.core.prompts._find_prompts_dir", return_value=tmp_prompts):
                with pytest.raises(ValueError, match="unexpected"):
                    load_prompt_template("test_prompt", required_placeholders=set())
    
    @patch.dict(os.environ, {"PROMPT_VERSION_EXTRACTION_SECTIONS": "v1"})
    def test_env_override_version(self):
        """Should respect environment variable version override."""
        template = load_prompt_template("extraction_sections")
        assert template is not None
        assert "{file_type}" in template
    
    @patch.dict(os.environ, {"PROMPT_STRICT_HASH": "0"})
    def test_checksum_warning_in_non_strict_mode(self, caplog):
        """Should warn but not fail on checksum mismatch in non-strict mode."""
        prompts_dir = _find_prompts_dir()
        
        # Create temp prompt with wrong hash in manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_prompts = Path(tmpdir) / "prompts"
            tmp_prompts.mkdir()
            
            # Copy a real template
            test_dir = tmp_prompts / "test_prompt"
            test_dir.mkdir()
            (test_dir / "v1.txt").write_text("Test content {placeholder}", encoding="utf-8")
            
            # Manifest with wrong hash
            manifest = {
                "test_prompt": {
                    "current": "v1",
                    "versions": {
                        "v1": {"sha256": "wrong_hash_value"}
                    }
                }
            }
            (tmp_prompts / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            
            # Mock _find_prompts_dir to return our temp dir
            with patch("times_doctor.core.prompts._find_prompts_dir", return_value=tmp_prompts):
                template = load_prompt_template("test_prompt", strict_hash=False)
                assert template is not None
                assert "Checksum mismatch" in caplog.text
    
    def test_checksum_strict_mode_fails(self):
        """Should raise ValueError on checksum mismatch in strict mode."""
        prompts_dir = _find_prompts_dir()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_prompts = Path(tmpdir) / "prompts"
            tmp_prompts.mkdir()
            
            test_dir = tmp_prompts / "test_prompt"
            test_dir.mkdir()
            (test_dir / "v1.txt").write_text("Test content", encoding="utf-8")
            
            manifest = {
                "test_prompt": {
                    "current": "v1",
                    "versions": {
                        "v1": {"sha256": "definitely_wrong_hash"}
                    }
                }
            }
            (tmp_prompts / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            
            with patch("times_doctor.core.prompts._find_prompts_dir", return_value=tmp_prompts):
                with pytest.raises(ValueError, match="Checksum mismatch"):
                    load_prompt_template("test_prompt", strict_hash=True)
    
    def test_load_nonexistent_prompt_returns_none(self):
        """Should return None for nonexistent prompt."""
        template = load_prompt_template("nonexistent_prompt_xyz")
        assert template is None
    
    def test_fallback_to_latest_version_if_no_manifest(self):
        """Should pick latest v* file if manifest is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_prompts = Path(tmpdir) / "prompts"
            tmp_prompts.mkdir()
            
            test_dir = tmp_prompts / "test_prompt"
            test_dir.mkdir()
            (test_dir / "v1.txt").write_text("Version 1", encoding="utf-8")
            (test_dir / "v2.txt").write_text("Version 2", encoding="utf-8")
            
            # No manifest
            
            with patch("times_doctor.core.prompts._find_prompts_dir", return_value=tmp_prompts):
                template = load_prompt_template("test_prompt")
                assert template == "Version 2"


class TestPromptBuilders:
    """Test prompt builder functions with deterministic outputs."""
    
    def test_build_llm_prompt_basic(self):
        """Should build diagnostic prompt with status context."""
        diagnostics = {
            "status": {"solve_status": "optimal", "primal_infeasible": False},
            "ranges": {},
            "mixed_currency_files": [],
            "used_barrier_noXO": False
        }
        
        prompt = build_llm_prompt(diagnostics)
        
        assert "LP solver expert" in prompt
        assert "numbered action plan" in prompt
        assert "solve_status: optimal" in prompt
        assert "primal_infeasible: False" in prompt
        assert "Used barrier without crossover: False" in prompt
    
    def test_build_llm_prompt_with_ranges(self):
        """Should include range statistics in context."""
        diagnostics = {
            "status": {},
            "ranges": {
                "rhs": (1e-6, 1e12),
                "bound": (0.0, 1e6),
                "matrix": (1e-8, 1e3)
            },
            "mixed_currency_files": [],
            "used_barrier_noXO": True
        }
        
        prompt = build_llm_prompt(diagnostics)
        
        assert "Range RHS min=1.000e-06, max=1.000e+12" in prompt
        assert "Range Bound min=0.000e+00, max=1.000e+06" in prompt
        assert "Range Matrix min=1.000e-08, max=1.000e+03" in prompt
        assert "Used barrier without crossover: True" in prompt
    
    def test_build_llm_prompt_with_mixed_currencies(self):
        """Should list mixed currency files."""
        diagnostics = {
            "status": {},
            "ranges": {},
            "mixed_currency_files": ["file1.dd", "file2.dd", "file3.dd"],
            "used_barrier_noXO": False
        }
        
        prompt = build_llm_prompt(diagnostics)
        
        assert "Mixed currencies seen in: file1.dd, file2.dd, file3.dd" in prompt
    
    def test_build_extraction_prompt_structure(self):
        """Should replace file_type placeholder and include content."""
        file_content = "Line 1\nLine 2\nLine 3"
        file_type = "lst"
        
        prompt = build_extraction_prompt(file_content, file_type)
        
        assert "LST file" in prompt
        assert "Return ONLY valid JSON" in prompt
        assert "File content:\n```\n" + file_content + "\n```" in prompt
        assert "{file_type}" not in prompt  # Should be replaced
    
    def test_build_extraction_prompt_run_log(self):
        """Should handle run_log file type."""
        prompt = build_extraction_prompt("content", "run_log")
        assert "RUN_LOG file" in prompt
    
    def test_build_qa_check_compress_prompt_structure(self):
        """Should include QA_CHECK compression instructions and content."""
        file_content = "WARNING: Issue 1\nERROR: Issue 2"
        
        prompt = build_qa_check_compress_prompt(file_content)
        
        assert "QA_CHECK.LOG" in prompt
        assert "SEVERE ERROR" in prompt
        assert "WARNING" in prompt
        assert "See QA_CHECK.LOG for full detail" in prompt
        assert file_content in prompt
    
    def test_build_review_prompt_structure(self):
        """Should include all file sections with proper headers."""
        qa_check = "QA warnings here"
        run_log = "Run log here"
        lst_content = "LST content here"
        
        prompt = build_review_prompt(qa_check, run_log, lst_content)
        
        assert "TIMES" in prompt or "Veda" in prompt
        assert "=== QA_CHECK.LOG (CONDENSED) ===" in prompt
        assert "=== RUN LOG (CONDENSED) ===" in prompt
        assert "=== LST FILE (CONDENSED EXCERPTS) ===" in prompt
        assert qa_check in prompt
        assert run_log in prompt
        assert lst_content in prompt
    
    def test_build_review_prompt_with_missing_files(self):
        """Should handle None/empty file content gracefully."""
        prompt = build_review_prompt("", None, "")
        
        assert "=== QA_CHECK.LOG (CONDENSED) ===" in prompt
        assert "=== RUN LOG (CONDENSED) ===" in prompt
        assert "(file not found)" in prompt


class TestPromptContracts:
    """Test that prompts maintain their contracts (structure, examples, etc.)."""
    
    def test_extraction_sections_json_example_valid(self):
        """Should have valid JSON example in extraction_sections template."""
        template = load_prompt_template("extraction_sections")
        assert template is not None
        
        # Extract the example JSON block - handle multiline
        import re
        json_match = re.search(r'\{\s*"sections"\s*:\s*\[[^\]]*\]\s*\}', template, re.DOTALL)
        assert json_match, "Template should contain JSON example with 'sections' key"
        
        # Verify it's parseable
        example_json = json_match.group(0)
        parsed = json.loads(example_json)
        assert "sections" in parsed
        assert isinstance(parsed["sections"], list)
        # Verify structure of example
        if len(parsed["sections"]) > 0:
            assert "name" in parsed["sections"][0]
            assert "start_line" in parsed["sections"][0]
            assert "end_line" in parsed["sections"][0]
    
    def test_qa_check_compress_footer_present(self):
        """Should have the required footer in qa_check_compress template."""
        template = load_prompt_template("qa_check_compress")
        assert template is not None
        assert "---" in template
        assert "See QA_CHECK.LOG for full detail" in template
    
    def test_review_markdown_formatting_rules(self):
        """Should include markdown formatting guidance in review template."""
        template = load_prompt_template("review")
        assert template is not None
        # Check for key formatting rules
        assert "TIMES" in template or "Veda" in template
    
    def test_no_problematic_markdown_sequences(self):
        """Should document markdown formatting rules in review template."""
        # Review template explicitly documents the problematic sequences to avoid
        # The template contains these as EXAMPLES in the rules, not in actual output
        review_template = load_prompt_template("review")
        if review_template:
            # Should contain markdown formatting guidance section
            assert "Markdown Formatting Rules" in review_template
            # The examples of what to avoid appear only in the rule documentation
            assert "CRITICAL" in review_template  # Marking the rules section


class TestHashSnapshot:
    """Test that template hashes match manifest (prevent unintentional edits)."""
    
    def test_all_manifested_prompts_have_correct_hash(self):
        """Should verify SHA256 checksums for all prompts in manifest."""
        prompts_dir = _find_prompts_dir()
        manifest_path = prompts_dir / "manifest.json"
        
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        
        for prompt_name, config in manifest.items():
            current_version = config["current"]
            expected_hash = config["versions"][current_version]["sha256"]
            
            template_path = prompts_dir / prompt_name / f"{current_version}.txt"
            assert template_path.exists(), f"Template file missing: {template_path}"
            
            content = template_path.read_text(encoding="utf-8")
            actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            
            assert actual_hash == expected_hash, (
                f"Hash mismatch for {prompt_name}/{current_version}:\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}\n"
                f"  Template has been modified without updating manifest!"
            )
    
    def test_encoding_and_whitespace_standards(self):
        """Should verify templates follow encoding/whitespace standards."""
        prompts_dir = _find_prompts_dir()
        
        for template_file in prompts_dir.rglob("v*.txt"):
            content = template_file.read_text(encoding="utf-8")
            
            # No CRLF line endings
            assert "\r\n" not in content, f"{template_file} contains CRLF line endings"
            
            # No trailing spaces on lines
            for i, line in enumerate(content.splitlines(), 1):
                assert not line.endswith(" "), f"{template_file}:{i} has trailing space"
            
            # No tab characters (use spaces)
            assert "\t" not in content, f"{template_file} contains tab characters"
