// ==============================================================================
// DyTrStudio Lightweight Pure-JavaScript ZIP Archive Builder
// Zero External Dependencies - RFC 1950 / RFC 1951 Compatible PKZip Construction
// Generates standard structured project archives for data, models, and code
// ==============================================================================

/**
 * CRC32 Checksum Table & Calculation
 */
const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c;
  }
  return table;
})();

function computeCRC32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC32_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Text encoding helper
 */
const textEncoder = new TextEncoder();

/**
 * Simple in-memory ZIP builder (Store compression mode 0)
 */
export class ZipBuilder {
  constructor() {
    this.files = [];
  }

  /**
   * Add a file to the ZIP archive
   * @param {string} path - Relative path inside zip (e.g. 'data/dataset.csv' or 'src/train.py')
   * @param {string|Uint8Array} content - File contents as string or binary Uint8Array
   */
  addFile(path, content) {
    const cleanPath = path.replace(/\\/g, '/').replace(/^\/+/, '');
    const data = typeof content === 'string' ? textEncoder.encode(content) : content;
    const crc = computeCRC32(data);
    this.files.push({
      path: cleanPath,
      data,
      crc,
      size: data.length,
    });
  }

  /**
   * Generate the ZIP Blob ready for browser download
   * @returns {Blob}
   */
  generateZipBlob() {
    const localHeaders = [];
    const centralDirectoryHeaders = [];
    let offset = 0;

    const date = new Date();
    const dosTime =
      (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1);
    const dosDate =
      ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();

    for (const file of this.files) {
      const pathBytes = textEncoder.encode(file.path);
      const localHeaderOffset = offset;

      // Local File Header (30 bytes + name + data)
      const localHeader = new Uint8Array(30 + pathBytes.length);
      const lv = new DataView(localHeader.buffer);

      lv.setUint32(0, 0x04034b50, true); // Local file header signature
      lv.setUint16(4, 20, true); // Version needed to extract (2.0)
      lv.setUint16(6, 0x0800, true); // General purpose bit flag (UTF-8)
      lv.setUint16(8, 0, true); // Compression method (0 = uncompressed / store)
      lv.setUint16(10, dosTime, true); // Last mod file time
      lv.setUint16(12, dosDate, true); // Last mod file date
      lv.setUint32(14, file.crc, true); // CRC-32
      lv.setUint32(18, file.size, true); // Compressed size
      lv.setUint32(22, file.size, true); // Uncompressed size
      lv.setUint16(26, pathBytes.length, true); // Filename length
      lv.setUint16(28, 0, true); // Extra field length
      localHeader.set(pathBytes, 30);

      localHeaders.push(localHeader);
      localHeaders.push(file.data);

      offset += localHeader.length + file.data.length;

      // Central Directory Header (46 bytes + name)
      const centralHeader = new Uint8Array(46 + pathBytes.length);
      const cv = new DataView(centralHeader.buffer);

      cv.setUint32(0, 0x02014b50, true); // Central directory file header signature
      cv.setUint16(4, 20, true); // Version made by
      cv.setUint16(6, 20, true); // Version needed to extract
      cv.setUint16(8, 0x0800, true); // General purpose bit flag (UTF-8)
      cv.setUint16(10, 0, true); // Compression method
      cv.setUint16(12, dosTime, true); // Last mod file time
      cv.setUint16(14, dosDate, true); // Last mod file date
      cv.setUint32(16, file.crc, true); // CRC-32
      cv.setUint32(20, file.size, true); // Compressed size
      cv.setUint32(24, file.size, true); // Uncompressed size
      cv.setUint16(28, pathBytes.length, true); // Filename length
      cv.setUint16(30, 0, true); // Extra field length
      cv.setUint16(32, 0, true); // File comment length
      cv.setUint16(34, 0, true); // Disk number start
      cv.setUint16(36, 0, true); // Internal file attributes
      cv.setUint32(38, 0, true); // External file attributes
      cv.setUint32(42, localHeaderOffset, true); // Relative offset of local header
      centralHeader.set(pathBytes, 46);

      centralDirectoryHeaders.push(centralHeader);
    }

    const centralDirectoryStart = offset;
    let centralDirectorySize = 0;
    for (const h of centralDirectoryHeaders) {
      centralDirectorySize += h.length;
    }

    // End of Central Directory Record (22 bytes)
    const eocd = new Uint8Array(22);
    const ev = new DataView(eocd.buffer);
    ev.setUint32(0, 0x06054b50, true); // End of central dir signature
    ev.setUint16(4, 0, true); // Number of this disk
    ev.setUint16(6, 0, true); // Disk where central directory starts
    ev.setUint16(8, this.files.length, true); // Number of central directory records on this disk
    ev.setUint16(10, this.files.length, true); // Total number of central directory records
    ev.setUint32(12, centralDirectorySize, true); // Size of central directory
    ev.setUint32(16, centralDirectoryStart, true); // Offset of start of central directory
    ev.setUint16(20, 0, true); // Comment length

    const allChunks = [...localHeaders, ...centralDirectoryHeaders, eocd];
    return new Blob(allChunks, { type: 'application/zip' });
  }

  /**
   * Directly trigger browser download of the ZIP
   * @param {string} filename
   */
  download(filename = 'dytr_scientific_bundle.zip') {
    const blob = this.generateZipBlob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}
