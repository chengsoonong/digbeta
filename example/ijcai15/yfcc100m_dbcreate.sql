/* create a database for YFCC100M dataset */
CREATE DATABASE yfcc100m;

/* create a table for all data records */
CREATE TABLE yfcc100m.tdata(
    /* Photo/video identifier */
    pv_id           BIGINT UNSIGNED NOT NULL UNIQUE PRIMARY KEY,
    /* User NSID */
    user_id         VARCHAR(64),
    /* User nickname */
    user_name       VARCHAR(64),
    /* Date taken */
    date_taken      DATETIME,
    /* Date uploaded */
    date_upload     DATETIME,
    /* Capture device */
    device          VARCHAR(64),
    /* Title */
    title           VARCHAR(64),
    /* Description */
    description     VARCHAR(256),
    /* User tags (comma-separated) */
    user_tags       VARCHAR(512),
    /* Machine tags (comma-separated) */
    machine_tags    VARCHAR(512),
    /* Longitude */
    longitude       VARCHAR(64),
    /* Latitude */
    latitude        VARCHAR(64),
    /* Accuracy */
    accuracy        VARCHAR(64),
    /* Photo/video page URL */
    pv_page_url     VARCHAR(256),
    /* Photo/video download URL */
    pv_download_url VARCHAR(256),
    /* License name */
    license_name    VARCHAR(64),
    /* License URL */
    license_url     VARCHAR(256),
    /* Photo/video server identifier */
    pv_server_id    VARCHAR(64),
    /* Photo/video farm identifier */
    pv_farm_id      VARCHAR(64),
    /* Photo/video secret */
    pv_secret       VARCHAR(64),
    /* Photo/video secret original */
    pv_secret_orig  VARCHAR(64),
    /* Photo/video extension original */ 
    pv_ext_orig     VARCHAR(8),
    /* Photos/video marker (0 = photo, 1 = video) */
    pv_marker       TINYINT(1)
);

/* commit transaction */
COMMIT;
