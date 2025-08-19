-- with d as (SELECT
--     uba.user_id,
--     books.isbn,
--     DATE(uba.created_at) as book_create_dt,
--     brf.name as language_book,
--     client_type,
--     platform,
--     CASE 
--         WHEN (EXTRACT(HOUR FROM (
--             CASE 
--                 WHEN b2b.browser_timezone IN ('us/pacific', 'us/eastern', 'us/central', 'us/mountain', 'utc',
--                 'us/arizona', 'us/hawaii', 'us/samoa', 'us/alaska', 'Asia/Kolkata', 'Europe/London', 
--                 'Asia/Shanghai', 'Australia/Sydney', 'Asia/Jakarta', 'Australia/Brisbane', 'Europe/Berlin', 
--                 'Asia/Baghdad', 'America/Bogota', 'Pacific/Auckland', 'Africa/Johannesburg', 'Australia/Perth', 
--                 'Asia/Karachi', 'Australia/Adelaide', 'Europe/Minsk', 'Europe/Helsinki', 'Africa/Cairo', 
--                 'America/Argentina/Buenos_Aires', 'Japan', 'Mexico/General', 'Canada/Newfoundland', 
--                 'Canada/Atlantic') 
--                 THEN convert_timezone('UTC', b2b.browser_timezone, uba.created_at) 
--                 ELSE uba.created_at 
--             END
--         )) NOT BETWEEN 4 AND 14) OR EXTRACT(DOW FROM uba.created_at) IN (0, 6) 
--         THEN 1 
--         ELSE 0 
--     END AS played_post_3pm,
--     row_number() over(partition by uba.user_id order by uba.created_at desc) as lasted_book_read
-- FROM
--     data_warehouse.student_klass_history sfl
-- JOIN exploratory_service.user_book_attempts uba
--     ON sfl.student_id = UPPER(REPLACE(uba.user_id, '-', ''))
--     AND sfl.school_year = 2025
--     AND DATE(uba.created_at) >= '2024-08-01'
-- JOIN book_service.books AS books ON uba.book_code = books.isbn
-- LEFT JOIN data_warehouse.dim_client AS client ON client.client_id = UPPER(REPLACE(uba.client_id, '-', '')) 
-- left join book_service.book_languages brf on brf.id = books.book_language_id
-- left join (select user_id,case when t.browser_timezone = 'arizona' then 'us/arizona'
--     		    when t.browser_timezone = 'hawaii' then 'us/hawaii'
-- 			    when t.browser_timezone = 'american samoa' then 'us/samoa'
-- 			    When t.browser_timezone = 'central america' then 'us/central'
-- 			    when t.browser_timezone = 'alaska' then 'us/alaska'
-- 			    when t.browser_timezone = 'edinburgh' then 'Europe/London'
-- 			    when t.browser_timezone = 'beijing' then 'Asia/Shangha'
-- 			    when t.browser_timezone = 'sydney' then 'Australia/Sydney'
-- 			    when t.browser_timezone = 'jakarta' then 'Asia/Jakarta'
-- 			    when t.browser_timezone = 'brisbane' then 'Australia/Brisbane'
-- 			    when t.browser_timezone = 'berlin' then 'Europe/Berlin'
-- 			    when t.browser_timezone = 'baghdad' then 'Asia/Baghdad'
-- 			    when t.browser_timezone = 'bogota' then 'America/Bogota'
-- 			    when t.browser_timezone = 'auckland' then 'Pacific/Auckland'
-- 			    when t.browser_timezone = 'pretoria' then 'Africa/Johannesburg'
-- 			    when t.browser_timezone = 'perth' then 'Australia/Perth'
-- 			    when t.browser_timezone = 'islamabad' then 'Asia/Karachi'
-- 			    when t.browser_timezone = 'adelaide' then 'Australia/Adelaide'
-- 			    when t.browser_timezone = 'minsk' then 'Europe/Minsk'
-- 			    when t.browser_timezone = 'helsinki' then 'Europe/Helsinki'
-- 			    when t.browser_timezone = 'cairo' then 'Africa/Cairo'
-- 			    when t.browser_timezone = 'buenos aires' then 'America/Argentina/Buenos_Aires'
-- 			    when t.browser_timezone = 'osaka' then 'Japan'
-- 			    when t.browser_timezone = 'guadalajara' then 'Mexico/General'
-- 			    when t.browser_timezone = 'newfoundland' then 'Canada/Newfoundland'
-- 			    when t.browser_timezone = 'atlantic time (canada)' then 'Canada/Atlantic'
-- 			    else t.browser_timezone end as browser_timezone	from data_warehouse.dim_customer t where user_type = 'teacher' )  b2b on sfl.klass_teacher_id = b2b.user_id 
-- )
-- ['web', 'apple', None, 'android']
-- select
--     user_id,
--     isbn,
--     max(book_create_dt) as book_create_dt,
--     sum(case when client_type = 'web' and played_post_3pm =0  then 1 else 0 end) as web_during_school_hour,
--     sum(case when client_type = 'web' and played_post_3pm =1  then 1 else 0 end) as web_after_school_hour,

--     sum(case when client_type = 'apple' and played_post_3pm =0  then 1 else 0 end) as apple_during_school_hour,
--     sum(case when client_type = 'apple' and played_post_3pm =1  then 1 else 0 end) as apple_after_school_hour,

--     sum(case when client_type = 'android' and played_post_3pm =0  then 1 else 0 end) as android_during_school_hour,
--     sum(case when client_type = 'android' and played_post_3pm =1  then 1 else 0 end) as android_after_school_hour,

--     sum(case when client_type = None and played_post_3pm =0  then 1 else 0 end) as unk_during_school_hour,
--     sum(case when client_type = None and played_post_3pm =1  then 1 else 0 end) as unk_after_school_hour,


-- from
-- d

with d as (SELECT
    uba.user_id,
    books.isbn,
    DATE(uba.created_at) as book_create_dt,
    brf.name as language_book,
    client_type,
    platform,
    CASE 
        WHEN (EXTRACT(HOUR FROM (
            CASE 
                WHEN b2b.browser_timezone IN ('us/pacific', 'us/eastern', 'us/central', 'us/mountain', 'utc',
                'us/arizona', 'us/hawaii', 'us/samoa', 'us/alaska', 'Asia/Kolkata', 'Europe/London', 
                'Asia/Shanghai', 'Australia/Sydney', 'Asia/Jakarta', 'Australia/Brisbane', 'Europe/Berlin', 
                'Asia/Baghdad', 'America/Bogota', 'Pacific/Auckland', 'Africa/Johannesburg', 'Australia/Perth', 
                'Asia/Karachi', 'Australia/Adelaide', 'Europe/Minsk', 'Europe/Helsinki', 'Africa/Cairo', 
                'America/Argentina/Buenos_Aires', 'Japan', 'Mexico/General', 'Canada/Newfoundland', 
                'Canada/Atlantic') 
                THEN convert_timezone('UTC', b2b.browser_timezone, uba.created_at) 
                ELSE uba.created_at 
            END
        )) NOT BETWEEN 4 AND 14) OR EXTRACT(DOW FROM uba.created_at) IN (0, 6) 
        THEN 1 
        ELSE 0 
    END AS played_post_3pm,
    row_number() over(partition by uba.user_id order by uba.created_at desc) as lasted_book_read
FROM
    data_warehouse.student_klass_history sfl
JOIN exploratory_service.user_book_attempts uba
    ON sfl.student_id = UPPER(REPLACE(uba.user_id, '-', ''))
    AND sfl.school_year = 2025
    AND DATE(uba.created_at) >= '2024-08-01'
JOIN book_service.books AS books ON uba.book_code = books.isbn
LEFT JOIN data_warehouse.dim_client AS client ON client.client_id = UPPER(REPLACE(uba.client_id, '-', '')) 
left join book_service.book_languages brf on brf.id = books.book_language_id
left join (select user_id,case when t.browser_timezone = 'arizona' then 'us/arizona'
    		    when t.browser_timezone = 'hawaii' then 'us/hawaii'
			    when t.browser_timezone = 'american samoa' then 'us/samoa'
			    When t.browser_timezone = 'central america' then 'us/central'
			    when t.browser_timezone = 'alaska' then 'us/alaska'
			    when t.browser_timezone = 'edinburgh' then 'Europe/London'
			    when t.browser_timezone = 'beijing' then 'Asia/Shangha'
			    when t.browser_timezone = 'sydney' then 'Australia/Sydney'
			    when t.browser_timezone = 'jakarta' then 'Asia/Jakarta'
			    when t.browser_timezone = 'brisbane' then 'Australia/Brisbane'
			    when t.browser_timezone = 'berlin' then 'Europe/Berlin'
			    when t.browser_timezone = 'baghdad' then 'Asia/Baghdad'
			    when t.browser_timezone = 'bogota' then 'America/Bogota'
			    when t.browser_timezone = 'auckland' then 'Pacific/Auckland'
			    when t.browser_timezone = 'pretoria' then 'Africa/Johannesburg'
			    when t.browser_timezone = 'perth' then 'Australia/Perth'
			    when t.browser_timezone = 'islamabad' then 'Asia/Karachi'
			    when t.browser_timezone = 'adelaide' then 'Australia/Adelaide'
			    when t.browser_timezone = 'minsk' then 'Europe/Minsk'
			    when t.browser_timezone = 'helsinki' then 'Europe/Helsinki'
			    when t.browser_timezone = 'cairo' then 'Africa/Cairo'
			    when t.browser_timezone = 'buenos aires' then 'America/Argentina/Buenos_Aires'
			    when t.browser_timezone = 'osaka' then 'Japan'
			    when t.browser_timezone = 'guadalajara' then 'Mexico/General'
			    when t.browser_timezone = 'newfoundland' then 'Canada/Newfoundland'
			    when t.browser_timezone = 'atlantic time (canada)' then 'Canada/Atlantic'
			    else t.browser_timezone end as browser_timezone	from data_warehouse.dim_customer t where user_type = 'teacher' )  b2b on sfl.klass_teacher_id = b2b.user_id 
),

f_data as (select
    user_id,
    isbn,
    max(book_create_dt) as book_create_dt,
    sum(case when client_type = 'web' and played_post_3pm =0  then 1 else 0 end) as web_during_school_hour,
    sum(case when client_type = 'web' and played_post_3pm =1  then 1 else 0 end) as web_after_school_hour,

    sum(case when client_type = 'apple' and played_post_3pm =0  then 1 else 0 end) as apple_during_school_hour,
    sum(case when client_type = 'apple' and played_post_3pm =1  then 1 else 0 end) as apple_after_school_hour,

    sum(case when client_type = 'android' and played_post_3pm =0  then 1 else 0 end) as android_during_school_hour,
    sum(case when client_type = 'android' and played_post_3pm =1  then 1 else 0 end) as android_after_school_hour,

    sum(case when client_type is Null and played_post_3pm =0  then 1 else 0 end) as unk_during_school_hour,
    sum(case when client_type is Null and played_post_3pm =1  then 1 else 0 end) as unk_after_school_hour


from
d
group by 1,2)

select
    user_id,
    isbn as book_code,
    book_create_dt,
    
    SUM(web_during_school_hour) OVER (
        PARTITION BY user_id 
        ORDER BY book_create_dt
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_web_during_school_hour,
    
    SUM(web_after_school_hour) OVER (
        PARTITION BY user_id 
        ORDER BY book_create_dt
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_web_after_school_hour,
    
    SUM(apple_during_school_hour) OVER (
        PARTITION BY user_id 
        ORDER BY book_create_dt
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_apple_during_school_hour,
    
    SUM(apple_after_school_hour) OVER (
        PARTITION BY user_id 
        ORDER BY book_create_dt
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_apple_after_school_hour,
    
    SUM(android_during_school_hour) OVER (
        PARTITION BY user_id 
        ORDER BY book_create_dt
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_android_during_school_hour,
    
    SUM(android_after_school_hour) OVER (
        PARTITION BY user_id 
        ORDER BY book_create_dt
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_android_after_school_hour,
    
    SUM(unk_during_school_hour) OVER (
        PARTITION BY user_id 
        ORDER BY book_create_dt
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_unk_during_school_hour,
    
    SUM(unk_after_school_hour) OVER (
        PARTITION BY user_id 
        ORDER BY book_create_dt
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_unk_after_school_hour
    
from
f_data
