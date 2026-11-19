	.syntax unified
	.thumb
	.text
	.global main
	.type main, %function
	.thumb_func
main:
.Lcode:
	movs r0, #0
	ldr r1, =slot
	str r0, [r1]
	movs r0, #1
	movs r0, #2
	movs r0, #3
	bx lr
	.size main, .-main

	.data
	.global slot
	.type slot, %object
	.size slot, 4
slot:
	.word 0

	.section .debug_abbrev,"",%progbits
.Labbrev:
	.uleb128 1
	.uleb128 0x11
	.byte 0
	.uleb128 0x10
	.uleb128 0x06
	.uleb128 0x03
	.uleb128 0x08
	.byte 0, 0
	.byte 0

	.section .debug_info,"",%progbits
	.4byte .Linfo_end - .Linfo_start
.Linfo_start:
	.2byte 3
	.4byte .Labbrev
	.byte 4
	.uleb128 1
	.4byte .Lline
	.asciz "line_rows.c"
.Linfo_end:
	.4byte .Linfo2_end - .Linfo2_start
.Linfo2_start:
	.2byte 3
	.4byte .Labbrev
	.byte 4
	.uleb128 1
	.4byte .Lline2
	.asciz "other.c"
.Linfo2_end:

	.section .debug_line,"",%progbits
.Lline:
	.4byte .Lline_end - .Lline_start
.Lline_start:
	.2byte 3
	.4byte .Lheader_end - .Lheader_start
.Lheader_start:
	.byte 1
	.byte 1
	.byte -5
	.byte 14
	.byte 13
	.byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1
	.byte 0
	.asciz "line_rows.c"
	.uleb128 0, 0, 0
	.asciz "elsewhere.c"
	.uleb128 0, 0, 0
	.byte 0
.Lheader_end:
	.byte 0, 5, 2
	.4byte .Lcode
	.byte 3
	.sleb128 2
	.byte 1
	.byte 2
	.uleb128 2
	.byte 3
	.sleb128 1
	.byte 1
	.byte 3
	.sleb128 1
	.byte 1
	.byte 2
	.uleb128 2
	.byte 3
	.sleb128 -5
	.byte 1
	.byte 2
	.uleb128 2
	.byte 4
	.uleb128 9
	.byte 3
	.sleb128 7
	.byte 1
	.byte 2
	.uleb128 2
	.byte 4
	.uleb128 1
	.byte 3
	.sleb128 1
	.byte 1
	.byte 3
	.sleb128 -8
	.byte 1
	.byte 2
	.uleb128 2
	.byte 3
	.sleb128 9
	.byte 1
	.byte 2
	.uleb128 2
	.byte 3
	.sleb128 1
	.byte 1
	.byte 2
	.uleb128 2
	.byte 0, 1, 1
	.byte 0, 5, 2
	.4byte .Lcode + 10
	.byte 3
	.sleb128 19
	.byte 1
	.byte 2
	.uleb128 2
	.byte 0, 1, 1
.Lline_end:
.Lline2:
	.4byte .Lline2_end - .Lline2_start
.Lline2_start:
	.2byte 3
	.4byte .Lheader2_end - .Lheader2_start
.Lheader2_start:
	.byte 1
	.byte 1
	.byte -5
	.byte 14
	.byte 13
	.byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1
	.byte 0
	.asciz "other.c"
	.uleb128 0, 0, 0
	.byte 0
.Lheader2_end:
	.byte 0, 5, 2
	.4byte .Lcode + 12
	.byte 3
	.sleb128 29
	.byte 1
	.byte 2
	.uleb128 2
	.byte 0, 1, 1
	.byte 0, 5, 2
	.4byte 0
	.byte 3
	.sleb128 39
	.byte 1
	.byte 9
	.2byte .Lcode + 20
	.byte 0, 1, 1
.Lline2_end:
