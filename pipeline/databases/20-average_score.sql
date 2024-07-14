-- script that creates a stored procedure ComputeAverageScoreForUser
-- that computes and store the average score for a student
DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser (
    IN user_id_param INT
)
BEGIN
    DECLARE user_avg_score FLOAT;

    SELECT AVG(score) INTO user_avg_score
    FROM corrections
    WHERE user_id = user_id_param;

    UPDATE users
    SET average_score = user_avg_score
    WHERE id = user_id_param;
END //

DELIMITER ;