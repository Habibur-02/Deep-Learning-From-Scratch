"""
Plot Data Augmentation Script for Medical/BioPharma Data - Google Colab Version
Only modifies text files, keeps original images unchanged
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re
from google.colab import drive

# Mount Google Drive (only needed once, but included for completeness)
drive.mount('/content/drive')

class MedicalPlotAugmenter:
    """
    Augments medical/bioPharma plot dataset by creating text file errors
    Designed for reasoning-based quality check models
    """

    def __init__(self, original_dataset_path: str, augmented_dataset_path: str):
        self.original_path = Path(original_dataset_path)
        self.augmented_path = Path(augmented_dataset_path)
        self.generated_samples = []

    def setup_directories(self):
        """Create the augmented dataset folder structure"""
        dirs = [
            self.augmented_path / "images",
            self.augmented_path / "text_data",
            self.augmented_path / "annotations"
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")

    def parse_text_file(self, txt_path: Path) -> Dict:
        """
        Parse original text file (supports both .txt and .json)
        Expected format:
        TXT: LineName: [1,2,3,4]
        JSON: {"LineName": [1,2,3,4]}
        """
        lines_data = {}

        try:
            with open(txt_path, 'r') as f:
                content = f.read().strip()

            # Try JSON first
            if txt_path.suffix == '.json' or content.startswith('{'):
                data = json.loads(content)
                if isinstance(data, dict):
                    lines_data = data
                elif isinstance(data, list):
                    # Handle array of objects format (like your example)
                    for item in data:
                        if 'label' in item and 'lineName' in item['label']:
                            line_name = item['label']['lineName']
                            # Extract points - handle different formats
                            if 'points' in item and isinstance(item['points'], list):
                                # Extract only numeric y-values (or x,y pairs)
                                points = []
                                for p in item['points']:
                                    if 'y' in p and p.get('label') not in ['ymin', 'ymax', 'xmin', 'xmax']:
                                        points.append(p['y'])
                                if points:
                                    lines_data[line_name] = points
            else:
                # Parse TXT format: LineName: [1,2,3,4]
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    if ':' in line:
                        parts = line.split(':', 1)
                        line_name = parts[0].strip()
                        data_str = parts[1].strip()

                        # Extract numbers
                        numbers = re.findall(r'-?\d+\.?\d*', data_str)
                        if numbers:
                            data_points = [float(n) for n in numbers]
                            lines_data[line_name] = data_points

        except Exception as e:
            print(f"Error parsing {txt_path}: {e}")

        return {
            "lines": lines_data,
            "raw_content": content,
            "file_extension": txt_path.suffix
        }

    def write_text_file(self, lines_data: Dict, output_path: Path, original_format: str = '.txt'):
        """Write lines data to text file in original format"""
        if original_format == '.json':
            with open(output_path, 'w') as f:
                json.dump(lines_data, f, indent=2)
        else:
            with open(output_path, 'w') as f:
                for line_name, points in lines_data.items():
                    f.write(f"{line_name}: {points}\n")

    # ==================== ERROR GENERATION FUNCTIONS ====================

    def error_line_name_missing(self, original_data: Dict, output_txt_path: Path) -> Dict:
        """Remove one complete line (name + all data points)"""
        lines = original_data["lines"].copy()
        if len(lines) <= 1:
            return None

        removed_line = random.choice(list(lines.keys()))
        del lines[removed_line]

        self.write_text_file(lines, output_txt_path, original_data.get("file_extension", '.txt'))

        return {
            "status": "error",
            "error_type": "line_name_missing",
            "details": f"Complete line '{removed_line}' is missing from text file",
            "missing_line": removed_line,
            "severity": "high"
        }

    def error_data_point_missing(self, original_data: Dict, output_txt_path: Path) -> Dict:
        """Remove one data point from a random line"""
        lines = original_data["lines"].copy()
        if not lines:
            return None

        line_name = random.choice(list(lines.keys()))
        points = lines[line_name].copy()

        if len(points) <= 1:
            return None

        removed_index = random.randint(0, len(points) - 1)
        removed_value = points[removed_index]
        points.pop(removed_index)
        lines[line_name] = points

        self.write_text_file(lines, output_txt_path, original_data.get("file_extension", '.txt'))

        return {
            "status": "error",
            "error_type": "data_point_missing",
            "details": f"Data point at position {removed_index + 1} (value {removed_value}) missing from '{line_name}'",
            "affected_line": line_name,
            "removed_value": removed_value,
            "removed_position": removed_index,
            "severity": "medium"
        }

    def error_extra_data_point(self, original_data: Dict, output_txt_path: Path) -> Dict:
        """Add an extra data point to a random line"""
        lines = original_data["lines"].copy()
        if not lines:
            return None

        line_name = random.choice(list(lines.keys()))
        points = lines[line_name].copy()

        if points:
            # Generate extra point based on existing pattern
            avg = sum(points) / len(points)
            std = max(points) - min(points) if len(points) > 1 else avg * 0.1
            extra_value = round(random.uniform(avg - std, avg + std), 2)
            insert_position = random.randint(0, len(points))
            points.insert(insert_position, extra_value)
            lines[line_name] = points

        self.write_text_file(lines, output_txt_path, original_data.get("file_extension", '.txt'))

        return {
            "status": "error",
            "error_type": "extra_data_point",
            "details": f"Extra data point ({extra_value}) added to '{line_name}' line at position {insert_position + 1}",
            "affected_line": line_name,
            "added_value": extra_value,
            "severity": "medium"
        }

    def error_wrong_value(self, original_data: Dict, output_txt_path: Path) -> Dict:
        """Change one data point to a wrong value"""
        lines = original_data["lines"].copy()
        if not lines:
            return None

        line_name = random.choice(list(lines.keys()))
        points = lines[line_name].copy()

        if points:
            position = random.randint(0, len(points) - 1)
            original_value = points[position]

            # Change to significantly different value
            if original_value != 0:
                # Can be much higher or lower
                if random.choice([True, False]):
                    new_value = round(original_value * random.uniform(1.5, 3.0), 2)
                else:
                    new_value = round(original_value * random.uniform(0.1, 0.6), 2)
            else:
                new_value = round(random.uniform(1, 100), 2)

            points[position] = new_value
            lines[line_name] = points

        self.write_text_file(lines, output_txt_path, original_data.get("file_extension", '.txt'))

        return {
            "status": "error",
            "error_type": "wrong_value",
            "details": f"Wrong value in '{line_name}' line: {original_value} → {new_value}",
            "affected_line": line_name,
            "original_value": original_value,
            "wrong_value": new_value,
            "position": position,
            "severity": "high"
        }

    def error_wrong_line_name_typo(self, original_data: Dict, output_txt_path: Path) -> Dict:
        """Change line name to typo (realistic medical term typos)"""
        lines = original_data["lines"].copy()
        if not lines:
            return None

        old_name = random.choice(list(lines.keys()))

        # Generate realistic medical term typos
        def generate_typo(name: str) -> str:
            # Common medical typo patterns
            typo_patterns = [
                lambda s: s.replace('c', 'k'),           # Placebo → Placeko
                lambda s: s.replace('e', 'a'),           # Exenatide → Exanatide
                lambda s: s.replace('i', 'y'),           # HbA1c → HbA1y
                lambda s: s.replace('o', 'u'),           # Placebo → Placebu
                lambda s: s.lower(),                     # all lowercase
                lambda s: s.upper(),                     # all uppercase
                lambda s: s.replace('_', ''),            # Remove underscore
                lambda s: s.replace('_', '-'),           # Underscore to dash
                lambda s: s[:-1] if len(s) > 3 else s,   # Remove last char
                lambda s: s + 'x' if len(s) > 3 else s,  # Add x at end
            ]
            pattern = random.choice(typo_patterns)
            return pattern(name)

        new_name = generate_typo(old_name)
        lines[new_name] = lines.pop(old_name)

        self.write_text_file(lines, output_txt_path, original_data.get("file_extension", '.txt'))

        return {
            "status": "error",
            "error_type": "wrong_line_name",
            "details": f"Line name typo: '{old_name}' written as '{new_name}'",
            "original_name": old_name,
            "wrong_name": new_name,
            "severity": "high"
        }

    def error_empty_file(self, original_data: Dict, output_txt_path: Path) -> Dict:
        """Text file completely empty"""
        with open(output_txt_path, 'w') as f:
            f.write("")

        return {
            "status": "error",
            "error_type": "empty_data_file",
            "details": "Text file is completely empty",
            "severity": "critical"
        }

    def error_format_error(self, original_data: Dict, output_txt_path: Path) -> Dict:
        """Corrupted format in text file"""
        lines = original_data["lines"].copy()

        corruption_type = random.choice([
            "missing_colon",     # "HbA1c_Placebo [6.5,7.2]" (no colon)
            "unclosed_bracket",  # "HbA1c_Placebo: [6.5,7.2" (missing ])
            "random_text",       # Add random text
            "broken_json",       # Remove quotes or brackets
        ])

        if corruption_type == "missing_colon":
            # Write without colon
            with open(output_txt_path, 'w') as f:
                for line_name, points in lines.items():
                    f.write(f"{line_name} {points}\n")

        elif corruption_type == "unclosed_bracket":
            with open(output_txt_path, 'w') as f:
                for line_name, points in lines.items():
                    points_str = str(points)
                    if points_str.endswith(']'):
                        points_str = points_str[:-1]
                    f.write(f"{line_name}: {points_str}\n")

        elif corruption_type == "random_text":
            with open(output_txt_path, 'w') as f:
                for line_name, points in lines.items():
                    f.write(f"{line_name}: {points}\n")
                f.write("RANDOM_CORRUPTION_!@#$%^&*\n")

        else:  # broken_json - only for JSON files
            if original_data.get("file_extension") == '.json':
                with open(output_txt_path, 'w') as f:
                    f.write('{"HbA1c_Placebo": [6.5,7.2,8.1],')  # Incomplete JSON
            else:
                # Fallback to missing colon
                with open(output_txt_path, 'w') as f:
                    for line_name, points in lines.items():
                        f.write(f"{line_name} {points}\n")

        return {
            "status": "error",
            "error_type": "format_error",
            "details": f"Text file has format error: {corruption_type}",
            "corruption_type": corruption_type,
            "severity": "critical"
        }

    def error_category_mismatch(self, original_data: Dict, output_txt_path: Path) -> Dict:
        """Text file shows different number of lines than image"""
        lines = original_data["lines"].copy()
        if len(lines) <= 1:
            return None

        # Remove one line to create mismatch
        removed_line = random.choice(list(lines.keys()))
        del lines[removed_line]

        self.write_text_file(lines, output_txt_path, original_data.get("file_extension", '.txt'))

        return {
            "status": "error",
            "error_type": "category_mismatch",
            "details": f"Text file shows {len(lines)} line(s) but image shows {len(original_data['lines'])} line(s)",
            "text_line_count": len(lines),
            "actual_line_count": len(original_data['lines']),
            "missing_category": removed_line,
            "severity": "high"
        }

    def error_line_order_wrong(self, original_data: Dict, output_txt_path: Path) -> Dict:
        """Shuffle the order of lines in text file"""
        lines = original_data["lines"].copy()
        line_names = list(lines.keys())

        if len(line_names) <= 1:
            return None

        random.shuffle(line_names)

        shuffled_lines = {}
        for ln in line_names:
            shuffled_lines[ln] = lines[ln]

        self.write_text_file(shuffled_lines, output_txt_path, original_data.get("file_extension", '.txt'))

        return {
            "status": "error",
            "error_type": "line_order_wrong",
            "details": f"Line order in text file is shuffled: {line_names}",
            "current_order": line_names,
            "severity": "low"
        }

    def error_data_points_swapped(self, original_data: Dict, output_txt_path: Path) -> Dict:
        """Swap data points between two lines"""
        lines = original_data["lines"].copy()
        if len(lines) < 2:
            return None

        line_names = list(lines.keys())
        line1 = random.choice(line_names)
        line2 = random.choice([l for l in line_names if l != line1])

        # Swap data points
        lines[line1], lines[line2] = lines[line2], lines[line1]

        self.write_text_file(lines, output_txt_path, original_data.get("file_extension", '.txt'))

        return {
            "status": "error",
            "error_type": "data_points_swapped",
            "details": f"Data points swapped between '{line1}' and '{line2}'",
            "line1": line1,
            "line2": line2,
            "severity": "high"
        }

    def error_partial_data_missing(self, original_data: Dict, output_txt_path: Path) -> Dict:
        """Remove last few data points from a line"""
        lines = original_data["lines"].copy()
        if not lines:
            return None

        line_name = random.choice(list(lines.keys()))
        points = lines[line_name].copy()

        if len(points) <= 2:
            return None

        # Remove last 1 or 2 points
        remove_count = random.randint(1, min(2, len(points) - 1))
        removed_values = points[-remove_count:]
        points = points[:-remove_count]
        lines[line_name] = points

        self.write_text_file(lines, output_txt_path, original_data.get("file_extension", '.txt'))

        return {
            "status": "error",
            "error_type": "partial_data_missing",
            "details": f"Last {remove_count} data point(s) missing from '{line_name}': {removed_values}",
            "affected_line": line_name,
            "removed_values": removed_values,
            "severity": "medium"
        }

    # ==================== MULTI-ERROR COMBINATIONS ====================

    def generate_multi_errors(self, original_data: Dict, output_txt_path: Path, errors_list: List[str]) -> Dict:
        """Apply multiple errors sequentially"""

        error_functions = {
            "line_name_missing": self.error_line_name_missing,
            "data_point_missing": self.error_data_point_missing,
            "wrong_value": self.error_wrong_value,
            "wrong_line_name": self.error_wrong_line_name_typo,
            "category_mismatch": self.error_category_mismatch,
            "data_points_swapped": self.error_data_points_swapped,
            "partial_data_missing": self.error_partial_data_missing,
        }

        # Create a working copy of the data
        current_data = original_data.copy()
        applied_errors = []

        # Use a temporary file for sequential modifications
        temp_path = output_txt_path.with_suffix('.temp' + output_txt_path.suffix)

        # Start with original data
        self.write_text_file(original_data["lines"], temp_path, original_data.get("file_extension", '.txt'))

        for error_name in errors_list:
            if error_name in error_functions:
                # Parse current temp file
                current_parsed = self.parse_text_file(temp_path)
                if current_parsed["lines"]:
                    result = error_functions[error_name](current_parsed, temp_path)
                    if result:
                        applied_errors.append(result)

        # Move temp to final location
        if temp_path.exists():
            shutil.move(str(temp_path), str(output_txt_path))

        if applied_errors:
            combined_details = "; ".join([e["details"] for e in applied_errors])
            error_types = [e["error_type"] for e in applied_errors]

            return {
                "status": "error",
                "error_type": "multiple_errors",
                "details": combined_details,
                "errors_combined": error_types,
                "error_count": len(applied_errors),
                "severity": "critical"
            }

        return None

    # ==================== MAIN AUGMENTATION PIPELINE ====================

    def augment_dataset(self, generate_single=True, generate_double=True, generate_triple=False):
        """Main function to generate augmented dataset"""

        print("📁 Setting up directories...")
        self.setup_directories()

        # Find all images and text files
        images_dir = self.original_path / "images"
        text_dir = self.original_path / "text_data"

        print(f"🔍 Looking for images in: {images_dir}")
        print(f"🔍 Looking for text files in: {text_dir}")

        if not images_dir.exists():
            print(f"❌ Error: {images_dir} not found!")
            print("Please check your folder structure.")
            return

        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))

        if not image_files:
            print(f"❌ No image files found in {images_dir}")
            return

        print(f"✅ Found {len(image_files)} image files")

        # Realistic error combinations for medical data
        realistic_double_errors = [
            ["line_name_missing", "data_point_missing"],
            ["wrong_value", "wrong_line_name"],
            ["data_point_missing", "wrong_value"],
            ["category_mismatch", "data_point_missing"],
            ["line_name_missing", "category_mismatch"],
            ["wrong_line_name", "data_points_swapped"],
            ["partial_data_missing", "wrong_value"],
        ]

        realistic_triple_errors = [
            ["line_name_missing", "data_point_missing", "wrong_value"],
            ["wrong_line_name", "wrong_value", "category_mismatch"],
            ["data_points_swapped", "partial_data_missing", "line_name_missing"],
        ]

        for idx, img_path in enumerate(image_files):
            # Get base name without extension
            base_name = img_path.stem

            # Find corresponding text file (could be .txt or .json)
            txt_path = None
            for ext in ['.txt', '.json']:
                candidate = text_dir / f"{base_name}{ext}"
                if candidate.exists():
                    txt_path = candidate
                    break

            if not txt_path:
                print(f"⚠️ Warning: No text file found for {img_path.name}")
                continue

            print(f"🔄 Processing [{idx+1}/{len(image_files)}]: {base_name}")

            # Parse original data
            original_data = self.parse_text_file(txt_path)
            if not original_data["lines"]:
                print(f"⚠️ Warning: Could not parse {txt_path}")
                continue

            # 1. Copy original
            original_img = self.augmented_path / "images" / f"{base_name}_original.png"
            original_txt = self.augmented_path / "text_data" / f"{base_name}_original{txt_path.suffix}"
            shutil.copy(img_path, original_img)
            shutil.copy(txt_path, original_txt)

            # Original annotation
            original_annotation = {
                "status": "correct",
                "error_type": "none",
                "details": "No issues found",
                "line_count": len(original_data["lines"]),
                "line_names": list(original_data["lines"].keys()),
                "severity": "none"
            }

            with open(self.augmented_path / "annotations" / f"{base_name}_original.json", 'w') as f:
                json.dump(original_annotation, f, indent=2)

            self.generated_samples.append({
                "image": f"{base_name}_original.png",
                "text": f"{base_name}_original{txt_path.suffix}",
                "annotation": original_annotation
            })

            # 2. Generate single errors
            if generate_single:
                error_configs = [
                    ("line_missing", self.error_line_name_missing),
                    ("point_missing", self.error_data_point_missing),
                    ("extra_point", self.error_extra_data_point),
                    ("wrong_value", self.error_wrong_value),
                    ("wrong_name", self.error_wrong_line_name_typo),
                    ("empty", self.error_empty_file),
                    ("format_error", self.error_format_error),
                    ("category_mismatch", self.error_category_mismatch),
                    ("order_wrong", self.error_line_order_wrong),
                    ("points_swapped", self.error_data_points_swapped),
                    ("partial_missing", self.error_partial_data_missing),
                ]

                for suffix, error_func in error_configs:
                    output_txt = self.augmented_path / "text_data" / f"{base_name}_{suffix}{txt_path.suffix}"
                    output_img = self.augmented_path / "images" / f"{base_name}_{suffix}.png"

                    shutil.copy(img_path, output_img)
                    annotation = error_func(original_data, output_txt)

                    if annotation:
                        with open(self.augmented_path / "annotations" / f"{base_name}_{suffix}.json", 'w') as f:
                            json.dump(annotation, f, indent=2)

                        self.generated_samples.append({
                            "image": f"{base_name}_{suffix}.png",
                            "text": f"{base_name}_{suffix}{txt_path.suffix}",
                            "annotation": annotation
                        })

            # 3. Generate double errors
            if generate_double:
                for i, error_combo in enumerate(realistic_double_errors):
                    suffix = f"double_{i+1}"
                    output_txt = self.augmented_path / "text_data" / f"{base_name}_{suffix}{txt_path.suffix}"
                    output_img = self.augmented_path / "images" / f"{base_name}_{suffix}.png"

                    shutil.copy(img_path, output_img)
                    annotation = self.generate_multi_errors(original_data, output_txt, error_combo)

                    if annotation:
                        with open(self.augmented_path / "annotations" / f"{base_name}_{suffix}.json", 'w') as f:
                            json.dump(annotation, f, indent=2)

                        self.generated_samples.append({
                            "image": f"{base_name}_{suffix}.png",
                            "text": f"{base_name}_{suffix}{txt_path.suffix}",
                            "annotation": annotation
                        })

            # 4. Generate triple errors (optional)
            if generate_triple:
                for i, error_combo in enumerate(realistic_triple_errors):
                    suffix = f"triple_{i+1}"
                    output_txt = self.augmented_path / "text_data" / f"{base_name}_{suffix}{txt_path.suffix}"
                    output_img = self.augmented_path / "images" / f"{base_name}_{suffix}.png"

                    shutil.copy(img_path, output_img)
                    annotation = self.generate_multi_errors(original_data, output_txt, error_combo)

                    if annotation:
                        with open(self.augmented_path / "annotations" / f"{base_name}_{suffix}.json", 'w') as f:
                            json.dump(annotation, f, indent=2)

                        self.generated_samples.append({
                            "image": f"{base_name}_{suffix}.png",
                            "text": f"{base_name}_{suffix}{txt_path.suffix}",
                            "annotation": annotation
                        })

        # Save summary
        self.save_summary()

    def save_summary(self):
        """Save summary report"""
        summary_path = self.augmented_path / "augmentation_summary.json"

        error_counts = {}
        for sample in self.generated_samples:
            error_type = sample["annotation"].get("error_type", "unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        summary = {
            "total_samples": len(self.generated_samples),
            "error_distribution": error_counts,
            "samples": self.generated_samples
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n AUGMENTATION COMPLETE!")
        print("="*50)
        print(f" Total samples generated: {len(self.generated_samples)}")
        print(f" Output directory: {self.augmented_path}")
        print(f" Summary saved: {summary_path}")
        print("\n Error type distribution:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {error_type}: {count}")
        print("="*50)


# ==================== RUN IN COLAB ====================

# YOUR PATH IN GOOGLE DRIVE
ORIGINAL_DATASET = "/content/drive/MyDrive/original_dataset"
AUGMENTED_DATASET = "/content/drive/MyDrive/augmented_dataset"

print(" Starting Data Augmentation for Medical Plot Dataset...")
print(f" Original dataset: {ORIGINAL_DATASET}")
print(f" Augmented dataset will be saved to: {AUGMENTED_DATASET}")
print()

# Check if original dataset exists
if not Path(ORIGINAL_DATASET).exists():
    print(f" Error: {ORIGINAL_DATASET} not found!")
    print("Please make sure your Google Drive is mounted and the path is correct.")
else:
    # Create augmenter
    augmenter = MedicalPlotAugmenter(ORIGINAL_DATASET, AUGMENTED_DATASET)

    # Run augmentation
    # generate_single=True: all single errors (11 types)
    # generate_double=True: realistic double error combinations
    # generate_triple=False: set True if you want triple errors (more data)
    augmenter.augment_dataset(
        generate_single=True,
        generate_double=True,
        generate_triple=False  # Set to True if you want more augmentation
    )

    # Optional: Print sample annotation
    print("\n📖 Sample annotation example:")
    sample_annotation_path = AUGMENTED_DATASET + "/annotations/"
    if Path(sample_annotation_path).exists():
        annotation_files = list(Path(sample_annotation_path).glob("*.json"))
        if annotation_files:
            with open(annotation_files[0], 'r') as f:
                sample = json.load(f)
                print(json.dumps(sample, indent=2)[:500] + "...")
