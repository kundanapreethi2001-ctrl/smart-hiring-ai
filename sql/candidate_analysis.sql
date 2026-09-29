CREATE DATABASE smart_hiring;
USE smart_hiring;
CREATE TABLE candidates (
    candidate_id VARCHAR(10) PRIMARY KEY,
    age INT,
    experience INT,
    cgpa DECIMAL(4,2),
    coding_score INT,
    python_score INT,
    sql_score INT,
    projects INT,
    internships INT,
    communication_score INT,
    shortlisted INT
);

SELECT COUNT(*) FROM candidates;

SELECT 
	AVG(cgpa) AS avg_cgpa
FROM candidates;

SELECT
	COUNT(*) FROM candidates
WHERE cgpa>8.5;

SELECT
	AVG(coding_score)
FROM candidates
WHERE shortlisted = 1;

SELECT
	COUNT(*) FROM candidates
GROUP BY shortlisted;

SELECT
	shortlisted,
	AVG(experience) AS avg_exp
FROM candidates
GROUP BY shortlisted;

SELECT 
	COUNT(*) FROM candidates
WHERE shortlisted=1 AND projects>2;

SELECT
	AVG(sql_score)
FROM candidates
GROUP BY shortlisted;

SELECT
	MAX(cgpa)
FROM candidates
WHERE shortlisted=1;

SELECT
	candidate_id
FROM candidates
WHERE python_score>(SELECT AVG(python_score) FROM candidates);

SELECT
	((SELECT COUNT(*) 
    FROM candidates 
    WHERE shortlisted=1) / COUNT(*)) * 100 
FROM candidates;

SELECT
	candidate_id,
    cgpa
FROM candidates
ORDER BY cgpa DESC 
LIMIT 5;

SELECT
	shortlisted,
	COUNT(*) AS number_of_candidates,
    AVG(cgpa) AS avg_cgpa,
    AVG(coding_score) AS avg_coding_score
FROM candidates
GROUP BY shortlisted;

























    








