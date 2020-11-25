--Main Table

CREATE TABLE public.main_table
(
    primary_id varchar(20) NOT NULL,
    name varchar(50),
    description text,
	cas_number varchar(20),
	unii varchar(20),
	state text,
	indication text,
	pharmacodynamics text,
	mechanism text,
	toxicity text,
	metabolism text,
	absorbtion text,
	halflife text,
	protein_binding text,
	route_of_elimination text,
	volume_of_distibuteion text,
	clearance text,
	classification xml,
	fda_lable text,
	msds text,
	reactions xml,
	snp_effects xml,
	
    CONSTRAINT main_table_pkey PRIMARY KEY (primary_id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.main_table
    OWNER to postgres;
	
--Secondary IDs Table

CREATE SEQUENCE public.secondary_ids_seq;

ALTER SEQUENCE public.secondary_ids_seq
    OWNER TO postgres;

CREATE TABLE public.secondary_ids_table
(
    pkey integer NOT NULL DEFAULT nextval('secondary_ids_seq'::regclass),
    primary_id varchar(20),
	secondary_id varchar(20),
	
    CONSTRAINT secondary_ids_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.secondary_ids_table
    OWNER to postgres;
	
--Groups
CREATE SEQUENCE public.groups_seq;

ALTER SEQUENCE public.groups_seq
    OWNER TO postgres;

CREATE TABLE public.groups_table
(
	pkey integer NOT NULL DEFAULT nextval('groups_seq'::regclass),
    primary_id varchar(20),
    groups text,
	
    CONSTRAINT groups_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.groups_table
    OWNER to postgres;
	
--Categories

CREATE SEQUENCE public.categories_seq;

ALTER SEQUENCE public.categories_seq
    OWNER TO postgres;

CREATE TABLE public.categories_table
(
	pkey integer NOT NULL DEFAULT nextval('categories_seq'::regclass),
    category text,
	
    CONSTRAINT categories_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.categories_table
    OWNER to postgres;

--CategoriesBridge

CREATE SEQUENCE public.categories_bridge_seq;

ALTER SEQUENCE public.categories_bridge_seq
    OWNER TO postgres;

CREATE TABLE public.categories_bridge_table
(
	pkey integer NOT NULL DEFAULT nextval('categories_bridge_seq'::regclass),
	drug_id varchar(20),
    cat_id integer,
	
    CONSTRAINT categories_bridge_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.categories_bridge_table
    OWNER to postgres;
	
--Dosages

CREATE SEQUENCE public.dosages_seq;

ALTER SEQUENCE public.dosages_seq
    OWNER TO postgres;

CREATE TABLE public.dosages_table
(
	pkey integer NOT NULL DEFAULT nextval('dosages_seq'::regclass),
	drug_id varchar(20),
    form text,
	route text,
	strength text,
	
    CONSTRAINT dosages_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.dosages_table
    OWNER to postgres;
	
--Synonyms

CREATE SEQUENCE public.synonyms_seq;

ALTER SEQUENCE public.synonyms_seq
    OWNER TO postgres;

CREATE TABLE public.synonyms_table
(
	pkey integer NOT NULL DEFAULT nextval('synonyms_seq'::regclass),
	drug_id varchar(20),
    synonym_name text,
	
    CONSTRAINT synonyms_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.synonyms_table
    OWNER to postgres;
	
--Products

CREATE SEQUENCE public.products_seq;

ALTER SEQUENCE public.products_seq
    OWNER TO postgres;

CREATE TABLE public.products_table
(
	pkey integer NOT NULL DEFAULT nextval('products_seq'::regclass),
	drug_id varchar(20),
    product_name text,
	labeller text,
	ndc_id text,
	ndc_product_code text,
	dpd_id text,
	ema_product_code text,
	ema_ma_number text,
	started_marketing_on date,
	ended_marketing_on date,
	dosage_form text,
	strength text,
	route text,
	fda_application_number text,
	generic  boolean,
	over_the_counter boolean,
	approved boolean,
	country varchar(20),
	source text,
	
    CONSTRAINT products_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.products_table
    OWNER to postgres;
	
--Mixtures

CREATE SEQUENCE public.mixtures_seq;

ALTER SEQUENCE public.mixtures_seq
    OWNER TO postgres;

CREATE TABLE public.mixtures_table
(
	pkey integer NOT NULL DEFAULT nextval('mixtures_seq'::regclass),
	drug_id varchar(20),
    mixture_name text,
	ingredient text,
	
    CONSTRAINT mixtures_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.mixtures_table
    OWNER to postgres;
	
--ATC CODES

CREATE SEQUENCE public.atc_codes_seq;

ALTER SEQUENCE public.atc_codes_seq
    OWNER TO postgres;

CREATE TABLE public.atc_codes_table
(
	pkey integer NOT NULL DEFAULT nextval('atc_codes_seq'::regclass),
	drug_id varchar(20),
    atc_codes xml,
	
    CONSTRAINT atc_codes_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.atc_codes_table
    OWNER to postgres;
	
--Drug Interactions

CREATE SEQUENCE public.drug_interactions_seq;

ALTER SEQUENCE public.drug_interactions_seq
    OWNER TO postgres;

CREATE TABLE public.drug_interactions_table
(
	pkey integer NOT NULL DEFAULT nextval('drug_interactions_seq'::regclass),
	drug_id_1 varchar(20),
    drug_id_2 varchar (20),
	sd_name text,
	sd_desc text,
	
    CONSTRAINT drug_interactions_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.drug_interactions_table
    OWNER to postgres;
	
--Sequences

CREATE SEQUENCE public.sequences_seq;

ALTER SEQUENCE public.sequences_seq
    OWNER TO postgres;

CREATE TABLE public.sequences_table
(
	pkey integer NOT NULL DEFAULT nextval('sequences_seq'::regclass),
	drug_id varchar(20),
    sequence_t text,
	
    CONSTRAINT sequences_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.sequences_table
    OWNER to postgres;
	
--Properties

CREATE SEQUENCE public.properties_seq;

ALTER SEQUENCE public.properties_seq
    OWNER TO postgres;

CREATE TABLE public.properties_table
(
	pkey integer NOT NULL DEFAULT nextval('properties_seq'::regclass),
	drug_id varchar(20),
    properties xml,
	
    CONSTRAINT properties_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.properties_table
    OWNER to postgres;
	
--Pathways

CREATE SEQUENCE public.pathways_seq;

ALTER SEQUENCE public.pathways_seq
    OWNER TO postgres;

CREATE TABLE public.pathways_table
(
	pkey integer NOT NULL DEFAULT nextval('pathways_seq'::regclass),
	drug_id varchar(20),
    pathways xml,
	
    CONSTRAINT pathways_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.pathways_table
    OWNER to postgres;
	
--Targets

CREATE SEQUENCE public.targets_seq;

ALTER SEQUENCE public.targets_seq
    OWNER TO postgres;

CREATE TABLE public.targets_table
(
	pkey integer NOT NULL DEFAULT nextval('targets_seq'::regclass),
	drug_id varchar(20),
    targets xml,
	
    CONSTRAINT targets_table_pkey PRIMARY KEY (pkey)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.targets_table
    OWNER to postgres;