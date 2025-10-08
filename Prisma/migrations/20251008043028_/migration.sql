-- CreateTable
CREATE TABLE "documents" (
    "id" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "insights" JSONB,
    "fullText" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,
    "document_type" TEXT,

    CONSTRAINT "documents_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "templates" (
    "id" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "file_description" TEXT,
    "jurisdiction" TEXT,
    "doc_type" TEXT,
    "similarity_tags" TEXT[],
    "body_md" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "templates_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "template_variables" (
    "id" TEXT NOT NULL,
    "key" TEXT NOT NULL,
    "label" TEXT NOT NULL,
    "description" TEXT,
    "example" TEXT,
    "required" BOOLEAN NOT NULL DEFAULT true,
    "template_id" TEXT NOT NULL,

    CONSTRAINT "template_variables_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "template_variables" ADD CONSTRAINT "template_variables_template_id_fkey" FOREIGN KEY ("template_id") REFERENCES "templates"("id") ON DELETE CASCADE ON UPDATE CASCADE;
