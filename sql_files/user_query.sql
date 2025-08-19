with book_engagement as 
(SELECT
    uba.book_code,
    uba.user_id,
    MAX(uba.created_at) book_create_dt,
    MAX(total_base) as total_pages,
    MAX(current_base) as max_read_pages
FROM
    data_warehouse.student_klass_history sfl
JOIN exploratory_service.user_book_attempts uba
    ON sfl.student_id = UPPER(REPLACE(uba.user_id, '-', ''))
    AND sfl.school_year = 2025
    AND DATE(uba.created_at) >= '2024-08-01'
GROUP BY 1,2),

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
book_format,
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
case when book_reading_format_id = 1 then 'RTM' when book_reading_format_id = 2 then 'RBM' end as book_type,
brf.name as language_book,
pub.name as publisher_name,
blt.name as fiction_nonfiction,
LISTAGG(distinct book_themes_t.name, ', ') theme_name,
LISTAGG(distinct book_categories_t.name,', ') category_name,
LISTAGG(DISTINCT rs.name, ', ') as reading_skill_name
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


select be.*,
row_number() over (partition by user_id ORDER by be.book_create_dt DESC) latest_to_old_rank,
books.theme_name,
books.category_name,
books.reading_skill_name,
books.language_book,
books.book_series
from book_engagement be
LEFT JOIN book_properties books on books.book_isbn = be.book_code
where books.id not in (5579,5578)