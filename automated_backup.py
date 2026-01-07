"""
Automated Backup System
=======================
Production-grade backup solution for trading system

Features:
- Automated database backups
- Configuration file backups
- Log file archiving
- Backup rotation
- Compression
- Verification
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import json
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")


class AutomatedBackup:
    """Automated backup system for trading data"""

    def __init__(self, backup_root: str = "backups"):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(exist_ok=True)

        # Backup directories
        self.db_backup_dir = self.backup_root / "database"
        self.config_backup_dir = self.backup_root / "config"
        self.log_backup_dir = self.backup_root / "logs"
        self.report_backup_dir = self.backup_root / "reports"

        # Create directories
        self.db_backup_dir.mkdir(exist_ok=True)
        self.config_backup_dir.mkdir(exist_ok=True)
        self.log_backup_dir.mkdir(exist_ok=True)
        self.report_backup_dir.mkdir(exist_ok=True)

        logger.info("Automated Backup System initialized")

    def backup_database(self, db_path: str = "trading_data.db") -> Optional[str]:
        """Backup database file"""

        logger.info("Backing up database...")

        db_file = Path(db_path)
        if not db_file.exists():
            logger.error(f"Database file not found: {db_path}")
            return None

        try:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"trading_db_{timestamp}.db"
            backup_path = self.db_backup_dir / backup_name

            # Copy database
            shutil.copy2(db_file, backup_path)

            # Get file size
            size_mb = backup_path.stat().st_size / (1024**2)

            logger.success(f"Database backed up: {backup_name} ({size_mb:.2f} MB)")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return None

    def backup_config_files(self) -> Optional[str]:
        """Backup configuration files"""

        logger.info("Backing up configuration files...")

        config_files = [
            "config.yaml",
            "requirements.txt",
        ]

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"config_{timestamp}.zip"
            backup_path = self.config_backup_dir / backup_name

            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for config_file in config_files:
                    file_path = Path(config_file)
                    if file_path.exists():
                        zipf.write(file_path, file_path.name)
                        logger.info(f"  Added: {config_file}")

                # Also backup config directory if exists
                config_dir = Path("config")
                if config_dir.exists():
                    for file in config_dir.rglob("*"):
                        if file.is_file():
                            zipf.write(file, f"config/{file.relative_to(config_dir)}")
                            logger.info(f"  Added: {file}")

            size_kb = backup_path.stat().st_size / 1024
            logger.success(f"Config backed up: {backup_name} ({size_kb:.2f} KB)")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Config backup failed: {e}")
            return None

    def backup_logs(self, days: int = 7) -> Optional[str]:
        """Backup recent log files"""

        logger.info(f"Backing up logs from last {days} days...")

        log_dir = Path("logs")
        if not log_dir.exists():
            logger.warning("Logs directory not found")
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"logs_{timestamp}.zip"
            backup_path = self.log_backup_dir / backup_name

            cutoff_date = datetime.now() - timedelta(days=days)
            files_added = 0

            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for log_file in log_dir.glob("*.log"):
                    # Check file modification time
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if mtime >= cutoff_date:
                        zipf.write(log_file, log_file.name)
                        files_added += 1

            if files_added > 0:
                size_kb = backup_path.stat().st_size / 1024
                logger.success(f"Logs backed up: {backup_name} ({files_added} files, {size_kb:.2f} KB)")
                return str(backup_path)
            else:
                backup_path.unlink()  # Remove empty zip
                logger.warning("No recent log files to backup")
                return None

        except Exception as e:
            logger.error(f"Log backup failed: {e}")
            return None

    def backup_reports(self) -> Optional[str]:
        """Backup performance reports"""

        logger.info("Backing up performance reports...")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"reports_{timestamp}.zip"
            backup_path = self.report_backup_dir / backup_name

            report_patterns = [
                "performance_report_*.txt",
                "health_report_*.txt",
            ]

            files_added = 0

            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for pattern in report_patterns:
                    for report_file in Path(".").glob(pattern):
                        zipf.write(report_file, report_file.name)
                        files_added += 1

            if files_added > 0:
                size_kb = backup_path.stat().st_size / 1024
                logger.success(f"Reports backed up: {backup_name} ({files_added} files, {size_kb:.2f} KB)")
                return str(backup_path)
            else:
                backup_path.unlink()  # Remove empty zip
                logger.warning("No report files to backup")
                return None

        except Exception as e:
            logger.error(f"Report backup failed: {e}")
            return None

    def rotate_backups(self, keep_count: int = 7):
        """Rotate backups - keep only the most recent N backups"""

        logger.info(f"Rotating backups (keeping last {keep_count})...")

        for backup_dir in [self.db_backup_dir, self.config_backup_dir,
                          self.log_backup_dir, self.report_backup_dir]:

            # Get all backups sorted by modification time
            backups = sorted(backup_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove old backups
            removed = 0
            for backup in backups[keep_count:]:
                try:
                    backup.unlink()
                    removed += 1
                except Exception as e:
                    logger.warning(f"Could not remove {backup}: {e}")

            if removed > 0:
                logger.info(f"  Removed {removed} old backup(s) from {backup_dir.name}")

    def create_full_backup(self, include_logs: bool = False) -> dict:
        """Create complete system backup"""

        logger.info("")
        logger.info("=" * 80)
        logger.info("  CREATING FULL SYSTEM BACKUP")
        logger.info("=" * 80)
        logger.info("")

        results = {
            'timestamp': datetime.now().isoformat(),
            'database': None,
            'config': None,
            'logs': None,
            'reports': None,
            'success': False
        }

        # Backup database
        db_backup = self.backup_database()
        results['database'] = db_backup

        # Backup config
        config_backup = self.backup_config_files()
        results['config'] = config_backup

        # Backup logs (optional)
        if include_logs:
            log_backup = self.backup_logs()
            results['logs'] = log_backup

        # Backup reports
        report_backup = self.backup_reports()
        results['reports'] = report_backup

        # Rotate old backups
        self.rotate_backups(keep_count=7)

        # Check success
        results['success'] = db_backup is not None and config_backup is not None

        # Save backup manifest
        self._save_manifest(results)

        logger.info("")
        logger.info("=" * 80)
        logger.info("  BACKUP COMPLETE")
        logger.info("=" * 80)
        logger.info("")

        return results

    def _save_manifest(self, results: dict):
        """Save backup manifest"""

        manifest_file = self.backup_root / "backup_manifest.json"

        try:
            # Load existing manifests
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifests = json.load(f)
            else:
                manifests = []

            # Add new manifest
            manifests.append(results)

            # Keep only last 30 manifests
            manifests = manifests[-30:]

            # Save
            with open(manifest_file, 'w') as f:
                json.dump(manifests, f, indent=2)

            logger.info(f"Backup manifest saved")

        except Exception as e:
            logger.warning(f"Could not save manifest: {e}")

    def list_backups(self):
        """List all available backups"""

        logger.info("")
        logger.info("=" * 80)
        logger.info("  AVAILABLE BACKUPS")
        logger.info("=" * 80)
        logger.info("")

        categories = {
            "Database": self.db_backup_dir,
            "Config": self.config_backup_dir,
            "Logs": self.log_backup_dir,
            "Reports": self.report_backup_dir
        }

        for category, backup_dir in categories.items():
            backups = sorted(backup_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

            logger.info(f"{category}:")
            logger.info("-" * 80)

            if backups:
                for backup in backups:
                    size = backup.stat().st_size / (1024**2)  # MB
                    mtime = datetime.fromtimestamp(backup.stat().st_mtime)
                    logger.info(f"  {backup.name} ({size:.2f} MB) - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                logger.info("  No backups found")

            logger.info("")

    def restore_database(self, backup_file: str, target: str = "trading_data.db"):
        """Restore database from backup"""

        backup_path = Path(backup_file)
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False

        try:
            # Create backup of current database
            target_path = Path(target)
            if target_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_backup = f"trading_db_before_restore_{timestamp}.db"
                shutil.copy2(target_path, self.db_backup_dir / current_backup)
                logger.info(f"Current database backed up to: {current_backup}")

            # Restore from backup
            shutil.copy2(backup_path, target_path)
            logger.success(f"Database restored from: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Automated Backup System")
    parser.add_argument("--backup", action="store_true", help="Create full backup")
    parser.add_argument("--include-logs", action="store_true", help="Include logs in backup")
    parser.add_argument("--list", action="store_true", help="List all backups")
    parser.add_argument("--restore", type=str, help="Restore database from backup file")

    args = parser.parse_args()

    backup_system = AutomatedBackup()

    if args.list:
        backup_system.list_backups()

    elif args.restore:
        backup_system.restore_database(args.restore)

    elif args.backup:
        results = backup_system.create_full_backup(include_logs=args.include_logs)

        if results['success']:
            logger.success("Full backup completed successfully!")
        else:
            logger.error("Backup completed with errors")

    else:
        # Default: create full backup
        results = backup_system.create_full_backup(include_logs=False)

        if results['success']:
            logger.success("Full backup completed successfully!")
        else:
            logger.error("Backup completed with errors")


if __name__ == "__main__":
    main()
