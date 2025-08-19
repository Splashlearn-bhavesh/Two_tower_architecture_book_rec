    with book_engagement as 
(SELECT
    uba.book_code,
    count(distinct uba.id) as clicks,
    count(distinct uba.user_id) as clicks_students,
    COUNT(DISTINCT CASE WHEN time_spent > 2 THEN uba.id ELSE NULL END) AS quality_clicks,
    COUNT(DISTINCT CASE WHEN time_spent > 2 THEN uba.user_id ELSE NULL END) AS quality_clicks_students,
    COUNT(DISTINCT CASE WHEN uba.completed_at IS NOT NULL THEN uba.user_id ELSE NULL END) AS students_completed_book,
    count(distinct case 
        when max_base > 0 and current_base / (max_base * 1.0) >= 0.75 
        then user_id 
    end)  as  students_completed_75_per_book,
  ROUND(
  (
    COUNT(DISTINCT CASE 
      WHEN max_base > 0 AND current_base / (max_base * 1.0) >= 0.75 
      THEN user_id 
    END
  )*100) / NULLIF(
    COUNT(DISTINCT CASE 
      WHEN time_spent > 0 THEN uba.user_id 
    END), 0), 2
) AS per_75_completed_unique_books,

    ROUND(
        (COUNT(DISTINCT CASE WHEN uba.completed_at IS NOT NULL THEN uba.user_id ELSE NULL END) * 100.0) /
        NULLIF(COUNT(DISTINCT CASE WHEN time_spent > 0 THEN uba.user_id ELSE NULL END), 0),
        2
    ) AS completion_rate,
    SUM(time_spent) AS time_spent,
    sum(total_base) as total_pages,
    sum(current_base) as read_pages
    -- LISTAGG(klass_grade_name, ', ') grade_name
FROM
    data_warehouse.student_klass_history sfl
JOIN exploratory_service.user_book_attempts uba
    ON sfl.student_id = UPPER(REPLACE(uba.user_id, '-', ''))
    AND sfl.school_year = 2025
    AND DATE(uba.created_at) >= '2024-08-01'
GROUP BY 1),

books_read_by_garde as
(select
book_code,
LISTAGG(klass_grade_name, ', ') grade_name
from
(select
uba.book_code,
klass_grade_name
FROM
    data_warehouse.student_klass_history sfl
JOIN exploratory_service.user_book_attempts uba
    ON sfl.student_id = UPPER(REPLACE(uba.user_id, '-', ''))
    AND sfl.school_year = 2025
    AND DATE(uba.created_at) >= '2024-08-01'
GROUP BY 1,2)
GROUP BY 1),

book_properties as 
(select 
distinct books.id as id,
isbn as book_isbn,
title as book_title,
authors,
bs.name book_series,
publication_date,
rights as rights,
illustrators,
interactive,
search_keywords,
top_hundred,
book_format as book_type,
books.long_description,
bestseller,
editor_recommended,
animated,
top_twenty,
top_fifty,
page_count,
min_grade,
max_grade,
readable_page_count,
min_reading_age,
max_reading_age,
read_along_audio,
read_along_with_highlighting,
orientation,
last_reading_page_number,
case when book_reading_format_id = 1 then 'RTM' when book_reading_format_id = 2 then 'RBM' end as book_format,
brf.name as language_book,
pub.name as publisher_name,
blt.name as fiction_nonfiction,
LISTAGG(DISTINCT rs.name, ', ') as reading_skill_name,
LISTAGG(DISTINCT book_themes_t.name, ', ') theme_name,
LISTAGG(DISTINCT book_categories_t.name,', ') category_name
from book_service.books
LEFT JOIN book_service.book_bisacs  AS book_bisacs_t ON book_bisacs_t.book_id=books.id
LEFT JOIN book_service.bisac_themes  AS bisac_themes_t ON book_bisacs_t.bisac_code_id=bisac_themes_t.bisac_code_id
LEFT JOIN book_service.book_themes  AS book_themes_t ON book_themes_t.id=bisac_themes_t.book_theme_id
LEFT JOIN book_service.book_categories  AS book_categories_t ON book_categories_t.id=bisac_themes_t.book_category_id
left join book_service.book_languages brf on brf.id = books.book_language_id
LEFT JOIN book_service.publishers pub on books.publisher_id = pub.id
LEFT JOIN book_service.book_literature_types blt on blt.id = books.book_literature_type_id
LEFT JOIN book_service.book_series_books bsb on bsb.book_id = books.id
LEFT JOIN book_service.book_series bs on bs.id = bsb.book_series_id
LEFT JOIN book_service.book_reading_skills brs on brs.book_id = books.id
LEFT JOIN book_service.reading_skills rs
on brs.reading_skill_id = rs.id

GROUP BY 
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
31, 32)


select * 
from book_properties books
left join books_read_by_garde brg on brg.book_code = books.id
left join book_engagement be on books.book_isbn = be.book_code
where books.id not in (5579,5578)


