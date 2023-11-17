import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddExifCityIndex1699746316571 implements MigrationInterface {
  name = 'AddExifCityIndex1699746316571';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE INDEX IF NOT EXISTS exif_city ON exif (city);`);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP INDEX IF EXISTS exif_city;`);
  }
}