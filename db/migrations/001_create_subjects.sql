-- SQL migration: create subjects table (SQLite compatible)
-- Save as db/migrations/001_create_subjects.sql

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS subjects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  semester INTEGER NOT NULL,
  subject_name TEXT NOT NULL,
  course_code TEXT,
  topics TEXT,
  credits INTEGER
);
