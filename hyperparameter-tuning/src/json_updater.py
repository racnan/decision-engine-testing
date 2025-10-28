"""
JSON constants file updater for hyperparameter tuning.

This module handles updating JSON constants files with new parameter values
during the optimization process, with backup safety mechanisms.
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any, Optional


class JSONUpdater:
    """Handles updating JSON constants files with new parameter values."""
    
    def __init__(self, file_path: str, section: str, backup_dir: Optional[str] = None):
        """
        Initialize the JSON updater.
        
        Args:
            file_path: Path to the JSON constants file
            section: Section name within the JSON to update
            backup_dir: Directory to store backup files (default: same directory as file)
        """
        self.file_path = file_path
        self.section = section
        self.backup_dir = backup_dir or os.path.dirname(file_path)
        
        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON constants file not found: {file_path}")
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self) -> str:
        """
        Create a backup of the current JSON file.
        
        Returns:
            Path to the backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(self.file_path)
        backup_name = f"{filename}.backup_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        shutil.copy2(self.file_path, backup_path)
        return backup_path
    
    def load_json(self) -> Dict[str, Any]:
        """
        Load the JSON file.
        
        Returns:
            Parsed JSON data
        """
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {self.file_path}: {e}")
        except Exception as e:
            raise IOError(f"Error reading file {self.file_path}: {e}")
    
    def save_json(self, data: Dict[str, Any], create_backup: bool = True) -> None:
        """
        Save data to the JSON file.
        
        Args:
            data: JSON data to save
            create_backup: Whether to create a backup before saving
        """
        if create_backup:
            self.create_backup()
        
        try:
            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise IOError(f"Error writing to file {self.file_path}: {e}")
    
    def update_parameters(self, parameters: Dict[str, Any], create_backup: bool = True) -> None:
        """
        Update specific parameters in the JSON file.
        
        Args:
            parameters: Dictionary of parameter names to values
            create_backup: Whether to create a backup before updating
        """
        # Load current JSON data
        data = self.load_json()
        
        # Navigate to the target section
        if self.section not in data:
            # Create section if it doesn't exist
            data[self.section] = {}
        
        target_section = data[self.section]
        
        # Update parameters
        for param_name, param_value in parameters.items():
            old_value = target_section.get(param_name, None)
            target_section[param_name] = param_value
            
            print(f"Updated {self.section}.{param_name}: {old_value} -> {param_value}")
        
        # Save updated data
        self.save_json(data, create_backup)
    
    def get_parameter(self, param_name: str) -> Any:
        """
        Get the current value of a parameter.
        
        Args:
            param_name: Name of the parameter to retrieve
            
        Returns:
            Current parameter value
        """
        data = self.load_json()
        
        if self.section not in data:
            return None
        
        return data[self.section].get(param_name, None)
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """
        Get all parameters in the target section.
        
        Returns:
            Dictionary of all parameters in the section
        """
        data = self.load_json()
        
        if self.section not in data:
            return {}
        
        return data[self.section].copy()
    
    def reset_to_backup(self, backup_path: str) -> None:
        """
        Reset the JSON file to a previous backup.
        
        Args:
            backup_path: Path to the backup file to restore from
        """
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        shutil.copy2(backup_path, self.file_path)
        print(f"Restored {self.file_path} from backup: {backup_path}")
    
    def list_backups(self) -> list:
        """
        List all backup files for this JSON file.
        
        Returns:
            List of backup file paths sorted by modification time (newest first)
        """
        filename = os.path.basename(self.file_path)
        backup_pattern = f"{filename}.backup_"
        
        backups = []
        for file in os.listdir(self.backup_dir):
            if file.startswith(backup_pattern):
                backup_path = os.path.join(self.backup_dir, file)
                backups.append(backup_path)
        
        # Sort by modification time (newest first)
        backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return backups
    
    def cleanup_old_backups(self, keep_count: int = 5) -> None:
        """
        Clean up old backup files, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent backups to keep
        """
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return
        
        # Remove oldest backups
        for backup_path in backups[keep_count:]:
            try:
                os.remove(backup_path)
                print(f"Removed old backup: {backup_path}")
            except Exception as e:
                print(f"Warning: Could not remove backup {backup_path}: {e}")
    
    def validate_section_structure(self) -> bool:
        """
        Validate that the target section exists and has the expected structure.
        
        Returns:
            True if section structure is valid, False otherwise
        """
        try:
            data = self.load_json()
            
            if self.section not in data:
                print(f"Warning: Section '{self.section}' not found in {self.file_path}")
                return False
            
            if not isinstance(data[self.section], dict):
                print(f"Warning: Section '{self.section}' is not a dictionary in {self.file_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating section structure: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation of the JSON updater."""
        return f"JSONUpdater(file={self.file_path}, section={self.section})"


class MultiFileUpdater:
    """Manages multiple JSON files for different algorithms."""
    
    def __init__(self):
        """Initialize the multi-file updater."""
        self.updaters = {}
    
    def add_updater(self, name: str, file_path: str, section: str, backup_dir: Optional[str] = None):
        """
        Add a JSON updater for a specific algorithm.
        
        Args:
            name: Name identifier for this updater
            file_path: Path to the JSON constants file
            section: Section name within the JSON to update
            backup_dir: Directory to store backup files
        """
        self.updaters[name] = JSONUpdater(file_path, section, backup_dir)
    
    def get_updater(self, name: str) -> JSONUpdater:
        """
        Get a specific updater by name.
        
        Args:
            name: Name of the updater to retrieve
            
        Returns:
            JSONUpdater instance
        """
        if name not in self.updaters:
            raise ValueError(f"No updater found with name: {name}")
        
        return self.updaters[name]
    
    def update_all(self, parameters_dict: Dict[str, Dict[str, Any]], create_backup: bool = True):
        """
        Update parameters in all configured JSON files.
        
        Args:
            parameters_dict: Dictionary mapping updater names to parameter dictionaries
            create_backup: Whether to create backups before updating
        """
        for updater_name, parameters in parameters_dict.items():
            if updater_name in self.updaters:
                self.updaters[updater_name].update_parameters(parameters, create_backup)
            else:
                print(f"Warning: No updater found for '{updater_name}', skipping")
    
    def cleanup_all_backups(self, keep_count: int = 5):
        """
        Clean up old backup files for all updaters.
        
        Args:
            keep_count: Number of recent backups to keep for each updater
        """
        for name, updater in self.updaters.items():
            print(f"Cleaning up backups for {name}...")
            updater.cleanup_old_backups(keep_count)
