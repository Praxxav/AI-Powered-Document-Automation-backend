-- AlterTable
ALTER TABLE "template_variables" ADD COLUMN     "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN     "enum" TEXT[],
ADD COLUMN     "regex" TEXT,
ADD COLUMN     "type" TEXT NOT NULL DEFAULT 'string';

-- CreateTable
CREATE TABLE "instances" (
    "id" TEXT NOT NULL,
    "template_id" TEXT NOT NULL,
    "user_query" TEXT NOT NULL DEFAULT '',
    "answers_json" TEXT NOT NULL,
    "draft_md" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "instances_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "instances" ADD CONSTRAINT "instances_template_id_fkey" FOREIGN KEY ("template_id") REFERENCES "templates"("id") ON DELETE CASCADE ON UPDATE CASCADE;
