-- SELECT
--     uba.user_id,
--    max(dim_location.country) country,
--    max(dim_location.state) state,
--    max(dim_location.zipcode) zipcode,
--    max(klass_grade_name) as klass_grade_name
-- FROM
--     data_warehouse.student_klass_history sfl
-- JOIN exploratory_service.user_book_attempts uba
--     ON sfl.student_id = UPPER(REPLACE(uba.user_id, '-', ''))
--     AND sfl.school_year = 2025
--     AND DATE(uba.created_at) >= '2024-08-01'
-- join data_warehouse.dim_customer dm on sfl.klass_teacher_id = dm.user_id
-- LEFT JOIN data_warehouse.dim_location  AS dim_location ON dm.user_id = dim_location.user_id
-- -- LEFT JOIN data_warehouse.dim_client AS client ON client.client_id = UPPER(REPLACE(uba.client_id, '-', '')) 
-- -- where UPPER(REPLACE(uba.user_id, '-', ''))= 'AEE540B30B3A4433A0E98E7557424183'
-- GROUP BY 1

SELECT 
*,
case when ac3 is not null then 'AC3'
    when ac2 is not null then 'AC2'
    when ac is not null then 'AC'
    when ac1 is not null then 'AC1'
    when ac0 is not null then 'AC0' else null end as class_activation_bucket

from
(SELECT
    uba.user_id,
    teacher_creation_source,
   max(dim_location.country) country,
   max(dim_location.state) state,
   max(dim_location.zipcode) zipcode,
   max(sfl.klass_grade_name) as klass_grade_name,
   max(sfl.klass_id) klass_id,
   max(sfl.classroom_type) classroom_type,
   max(sfl.klass_teacher_id) teacher_id,
   max(dm.create_dt) teacher_create_dt,
   max(case when date_part(month,(DATE(dm.create_dt ))) between 8 and 12 then date_part(year,(DATE(dm.create_dt )))+1 else date_part(year,(DATE(dm.create_dt ))) end) teacher_create_school_year,
   max(school_id) school_id,
   MAX(ac3_activation_dt) ac3,
   max(ac2_activation_dt) ac2,
   max(ac_activation_dt) ac,
   max(ac1_activation_dt) ac1,
   max(ac0_activation_dt) ac0
   
  
FROM
    data_warehouse.student_klass_history sfl
JOIN exploratory_service.user_book_attempts uba
    ON sfl.student_id = UPPER(REPLACE(uba.user_id, '-', ''))
    AND sfl.school_year = 2025
    AND DATE(uba.created_at) >= '2024-08-01'
join data_warehouse.dim_customer dm on sfl.klass_teacher_id = dm.user_id
LEFT JOIN data_warehouse.dim_location  AS dim_location ON dm.user_id = dim_location.user_id
-- LEFT JOIN data_warehouse.dim_client AS client ON client.client_id = UPPER(REPLACE(uba.client_id, '-', '')) 
-- where UPPER(REPLACE(uba.user_id, '-', ''))= 'AEE540B30B3A4433A0E98E7557424183'
where sfl.current_flag = 1
GROUP BY 1,2)