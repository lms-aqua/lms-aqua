package com.lostmediastudios.lostcam

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * JVM unit tests, so CI catches the framing and conversion mistakes without a
 * device. These are the same invariants the iOS tests assert, deliberately: the
 * two senders must be indistinguishable to a consumer.
 */
class HttpRequestTest {
    @Test
    fun `parses a simple request`() {
        val request = HttpRequest.parse("GET /video HTTP/1.1\r\nHost: phone:4747")
        assertEquals("GET", request?.method)
        assertEquals("/video", request?.path)
        assertEquals("phone:4747", request?.headers?.get("host"))
    }

    @Test
    fun `header names are case insensitive`() {
        val request = HttpRequest.parse("GET / HTTP/1.1\r\nX-LostCam-Token: abc")
        assertEquals("abc", request?.headers?.get("x-lostcam-token"))
        assertEquals("abc", request?.token)
    }

    @Test
    fun `parses query parameters`() {
        val request = HttpRequest.parse("GET /video?w=1280&h=720&q=70 HTTP/1.1")
        assertEquals("/video", request?.path)
        assertEquals("1280", request?.query?.get("w"))
        assertEquals("70", request?.query?.get("q"))
    }

    @Test
    fun `percent decodes query values`() {
        val request = HttpRequest.parse("GET /data?ch=attitude%2Cmotion HTTP/1.1")
        assertEquals("attitude,motion", request?.query?.get("ch"))
    }

    @Test
    fun `plus is decoded as a space`() {
        val request = HttpRequest.parse("GET /x?note=a+b HTTP/1.1")
        assertEquals("a b", request?.query?.get("note"))
    }

    @Test
    fun `trailing slash is normalised but root is preserved`() {
        assertEquals("/video", HttpRequest.parse("GET /video/ HTTP/1.1")?.path)
        assertEquals("/", HttpRequest.parse("GET / HTTP/1.1")?.path)
    }

    @Test
    fun `tolerates bare newlines`() {
        val request = HttpRequest.parse("GET /info HTTP/1.1\nHost: x")
        assertEquals("/info", request?.path)
        assertEquals("x", request?.headers?.get("host"))
    }

    @Test
    fun `rejects garbage rather than throwing`() {
        assertNull(HttpRequest.parse(""))
        assertNull(HttpRequest.parse("nonsense"))
    }

    @Test
    fun `int parameter clamps into range`() {
        val request = HttpRequest.parse("GET /data?hz=9999 HTTP/1.1")!!
        assertEquals(240, request.int("hz", 60, 1, 240))
    }

    @Test
    fun `int parameter falls back when unparseable`() {
        val request = HttpRequest.parse("GET /data?hz=fast HTTP/1.1")!!
        assertEquals(60, request.int("hz", 60, 1, 240))
    }

    @Test
    fun `routing covers every endpoint`() {
        assertEquals(LostCamServer.Route.STATUS, LostCamServer.routeFor("/"))
        assertEquals(LostCamServer.Route.INFO, LostCamServer.routeFor("/info"))
        assertEquals(LostCamServer.Route.VIDEO, LostCamServer.routeFor("/video"))
        assertEquals(LostCamServer.Route.AUDIO, LostCamServer.routeFor("/audio"))
        assertEquals(LostCamServer.Route.DATA, LostCamServer.routeFor("/data"))
        assertEquals(LostCamServer.Route.NOT_FOUND, LostCamServer.routeFor("/nope"))
    }
}

class MjpegFramingTest {
    @Test
    fun `content type carries the boundary`() {
        assertEquals(
            "multipart/x-mixed-replace; boundary=lostcamframe",
            MjpegFraming.streamContentType(),
        )
    }

    @Test
    fun `part header declares length and timestamp`() {
        val header = String(MjpegFraming.partHeader(1234, 5678))
        assertTrue(header.startsWith("--lostcamframe\r\n"))
        assertTrue(header.contains("Content-Length: 1234\r\n"))
        assertTrue(header.contains("X-LostCam-Timestamp: 5678\r\n"))
        assertTrue(header.endsWith("\r\n\r\n"))
    }

    @Test
    fun `part wraps the payload and declares its real length`() {
        val jpeg = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 1, 0xFF.toByte(), 0xD9.toByte())
        val part = MjpegFraming.part(jpeg, 1)
        val text = String(part)
        // A wrong length desynchronises every consumer using Content-Length.
        assertTrue(text.contains("Content-Length: 5"))
        assertTrue(text.endsWith("\r\n"))
    }
}

class HttpResponseTest {
    @Test
    fun `stream head has no content length`() {
        val head = String(HttpResponse.streamHead("x/y"))
        assertTrue(head.startsWith("HTTP/1.1 200 OK\r\n"))
        assertFalse(head.contains("Content-Length"))
        assertTrue(head.contains("Cache-Control: no-store"))
    }

    @Test
    fun `text response length matches the body`() {
        val response = String(HttpResponse.text(404, "Not Found", "nope"))
        assertTrue(response.contains("Content-Length: 4"))
        assertTrue(response.endsWith("nope"))
    }

    @Test
    fun `unauthorized is 401`() {
        assertTrue(String(HttpResponse.unauthorized).startsWith("HTTP/1.1 401"))
    }
}

class JsonWriterTest {
    @Test
    fun `writes a flat object`() {
        val json = JsonWriter().add("a", 1).add("b", "two").add("c", true).jsonString()
        assertEquals("""{"a":1,"b":"two","c":true}""", json)
    }

    @Test
    fun `finish appends a newline for NDJSON`() {
        assertEquals("{\"a\":1}\n", String(JsonWriter().add("a", 1).finish()))
    }

    @Test
    fun `map keys are sorted`() {
        val json = JsonWriter()
            .addMap("m", mapOf("b" to 0.5f, "a" to 0.25f))
            .jsonString()
        assertTrue(json.indexOf("\"a\"") < json.indexOf("\"b\""))
    }

    @Test
    fun `escapes quotes and control characters`() {
        val json = JsonWriter().add("device", "a \"quoted\"\nname").jsonString()
        assertTrue(json.contains("\\\""))
        assertTrue(json.contains("\\n"))
    }

    @Test
    fun `non finite numbers become zero rather than invalid JSON`() {
        // JSON has no NaN or Infinity; emitting them produces an unparseable doc.
        assertEquals("0", JsonWriter.number(Double.NaN, 3))
        assertEquals("0", JsonWriter.number(Double.POSITIVE_INFINITY, 3))
    }

    @Test
    fun `output parses as JSON`() {
        val bytes = JsonWriter()
            .add("t", 1000L)
            .add("ch", "motion")
            .addFloats("accel", floatArrayOf(0.1f, 0.2f, 0.98f))
            .finish()
        val parsed = JSONObject(String(bytes).trim())
        assertEquals("motion", parsed.getString("ch"))
        assertEquals(3, parsed.getJSONArray("accel").length())
    }
}

class MathsTest {
    @Test
    fun `identity quaternion is level`() {
        val angles = Maths.euler(0.0, 0.0, 0.0, 1.0)
        assertTrue(abs(angles[0]) < 1e-6)
        assertTrue(abs(angles[1]) < 1e-6)
        assertTrue(abs(angles[2]) < 1e-6)
    }

    @Test
    fun `ninety degrees about y is yaw`() {
        val half = (Math.PI / 2) / 2
        val angles = Maths.euler(0.0, sin(half), 0.0, cos(half))
        assertEquals(90.0, angles[1], 1e-4)
    }

    @Test
    fun `gimbal lock clamps instead of producing NaN`() {
        val half = (Math.PI / 2) / 2
        val angles = Maths.euler(0.0, sin(half), 0.0, cos(half))
        for (value in angles) assertFalse(value.isNaN())
    }

    @Test
    fun `acceleration is converted to g`() {
        // The whole point: Android reports m/s^2, the wire format is g. One place.
        assertEquals(1.0f, Maths.toG(9.80665f), 1e-5f)
        assertEquals(0.0f, Maths.toG(0f), 1e-6f)
    }

    @Test
    fun `pressure is converted to kilopascals`() {
        // 1013.25 hPa is standard atmosphere, which is 101.325 kPa.
        assertEquals(101.325, Maths.hectopascalsToKilopascals(1013.25f), 1e-3)
    }

    @Test
    fun `frame interval matches the requested rate`() {
        assertEquals(33L, Maths.frameIntervalMillis(30))
        assertEquals(0L, Maths.frameIntervalMillis(0))
    }

    @Test
    fun `quality and fps are clamped to usable ranges`() {
        assertEquals(100, Maths.jpegQuality(500))
        assertEquals(10, Maths.jpegQuality(-5))
        assertEquals(60, Maths.clampFps(999))
        assertEquals(1, Maths.clampFps(0))
    }
}
